"""Apply fail-closed, version-specific laboratory instrumentation."""
from pathlib import Path
import argparse
import subprocess

REVISION = "7dfd9a39649700c24c22f1978ed7a35541a72cca"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Upstream anchor changed: {old[:100]!r}")
    return text.replace(old, new, 1)


def patch(root):
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise RuntimeError(f"Unreviewed Open5GS revision: {actual}")
    here = Path(__file__).resolve().parent
    server = root / "lib/sbi/nghttp2-server.c"
    text = server.read_text()
    text = replace_once(text, "    bool                    memory_overflow;",
                        "    bool                    memory_overflow;\n    ogs_time_t lab_started;")
    text = replace_once(text, "} ogs_sbi_stream_t;", "} ogs_sbi_stream_t;\n" + (here / "server.inc").read_text())
    text = replace_once(text, "    nvlen = 3; /* :status && server && date */",
                        "    lab_record_response(stream, response->status);\n    nvlen = 3; /* :status && server && date */")
    text = replace_once(text, "            if (server->cb(request,",
                        "            if (lab_kie_request(stream)) return 0;\n\n            if (server->cb(request,")
    server.write_text(text)
    client = root / "lib/sbi/client.c"
    text = client.read_text()
    anchor = "static connection_t *connection_add("
    if text.count(anchor) != 2:
        raise RuntimeError("Client declaration/definition anchors changed")
    text = text.replace(anchor, (here / "client.inc").read_text() + "\n" + anchor, 1)
    text = replace_once(text, "            res = resource->data.result;",
                        "            lab_observe(easy, url, res_status);\n            res = resource->data.result;")
    client.write_text(text)
    main = root / "tests/registration/abts-main.c"
    text = main.read_text()
    text = replace_once(text, "    for (i = 0; alltests[i].func; i++)\n        suite = alltests[i].func(suite);",
                        """    {
        int round, rounds = getenv("LAB_ROUNDS") ? atoi(getenv("LAB_ROUNDS")) : 1;
        int gap = getenv("LAB_GAP_MS") ? atoi(getenv("LAB_GAP_MS")) : 0;
        ogs_time_t lab_start = ogs_time_now();
        for (round = 0; round < rounds; round++) {
            for (i = 0; alltests[i].func; i++) suite = alltests[i].func(suite);
            if (gap > 0) ogs_msleep(gap);
        }
        if (getenv("LAB_WORKLOAD_TIMING")) {
            FILE *timing = fopen(getenv("LAB_WORKLOAD_TIMING"), "w");
            if (timing) {
                fprintf(timing, "%lld", (long long)(ogs_time_now() - lab_start));
                fclose(timing);
            }
        }
    }""")
    main.write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    patch(parser.parse_args().root)
