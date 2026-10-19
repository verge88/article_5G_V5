/* Experimental instrumentation for the pinned Open5GS laboratory build.
 * Included after ogs_sbi_stream_t. Not a standardized 3GPP API.
 * Disabled unless LAB_KIE_KEY is set. Only loopback listeners are supported.
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */
#include <ctype.h>
#include <unistd.h>

static unsigned long long lab_requests, lab_responses, lab_errors, lab_service_us;
static unsigned long long lab_prev_requests, lab_prev_responses, lab_prev_errors, lab_prev_us;
static ogs_time_t lab_begin;
static unsigned long long lab_last_seq;

static void lab_hex(const uint8_t *in, char *out)
{
    int i;
    for (i = 0; i < 32; i++) sprintf(out + i * 2, "%02x", in[i]);
    out[64] = 0;
}

static int lab_mode(const char *ip)
{
    char path[1024];
    const char *dir = getenv("LAB_CONTROL_DIR");
    FILE *f;
    int mode = 0;
    if (!dir) return 0;
    snprintf(path, sizeof(path), "%s/%s", dir, ip);
    f = fopen(path, "r");
    if (f) { if (fscanf(f, "%d", &mode) != 1) mode = 0; fclose(f); }
    return mode;
}

static int lab_kie_request(ogs_sbi_stream_t *stream)
{
    const char *master = getenv("LAB_KIE_KEY");
    ogs_sbi_request_t *req = stream->request;
    char ip[OGS_ADDRSTRLEN], payload[1024], expected[65], mac[65];
    char body[1400];
    const char *seq, *nonce, *auth;
    uint8_t root_key[32], key[32], digest[32];
    ogs_time_t now;
    unsigned long long n, r, e, us;
    ogs_sbi_response_t *response;
    int mode, i, mismatch = 0;
    if (!master) return 0;
    OGS_ADDR(stream->session->server->node.addr, ip);
    if (strncmp(ip, "127.", 4)) return 0;
    if (!lab_begin) lab_begin = ogs_time_now();
    if (strcmp(req->h.uri, "/_kie/v1/report")) {
        stream->lab_started = ogs_time_now();
        lab_requests++;
        /* Benign scheduling delay: laboratory control, not exploit traffic. */
        if (lab_mode(ip) == 4) ogs_msleep(3);
        return 0;
    }
    /* The SCP must forward the KIE to the requested NF along its normal route. */
    if (!strcmp(ip, "127.0.0.200") &&
            ogs_sbi_header_get(req->http.headers, "3gpp-sbi-target-apiroot"))
        return 0;
    seq = ogs_sbi_header_get(req->http.headers, "x-kie-seq");
    nonce = ogs_sbi_header_get(req->http.headers, "x-kie-nonce");
    auth = ogs_sbi_header_get(req->http.headers, "x-kie-auth");
    if (!seq || !nonce || !auth || strlen(seq) > 16 ||
            strlen(nonce) != 32 || strlen(auth) != 64) goto forbidden;
    for (i = 0; seq[i]; i++) if (!isdigit((unsigned char)seq[i])) goto forbidden;
    for (i = 0; nonce[i]; i++) if (!isxdigit((unsigned char)nonce[i])) goto forbidden;
    /* Keep the cryptographic API input at its native 32-byte key size. */
    ogs_sha256((const uint8_t *)master, strlen(master), root_key);
    ogs_hmac_sha256(root_key, 32,
            (const uint8_t *)ip, strlen(ip), key, 32);
    snprintf(payload, sizeof(payload), "%s|%s|%s", ip, seq, nonce);
    ogs_hmac_sha256(key, 32, (uint8_t *)payload, strlen(payload), digest, 32);
    lab_hex(digest, expected);
    for (i = 0; i < 64; i++) mismatch |= (expected[i] ^ auth[i]);
    if (mismatch || !seq[0] || strtoull(seq, NULL, 10) <= lab_last_seq) goto forbidden;
    lab_last_seq = strtoull(seq, NULL, 10);

    now = ogs_time_now();
    n = lab_requests; r = lab_responses; e = lab_errors; us = lab_service_us;
    mode = lab_mode(ip);
    /* Fault injection changes telemetry only and then authenticates it correctly.
     * The independent SCP observer is never changed by these controls. */
    if (mode == 1) { n = n * 3 / 4; r = r * 3 / 4; e = e * 3 / 4; us = us * 3 / 4; }
    if (mode == 2) us /= 2;
    if (mode == 3) {
        n = lab_prev_requests; r = lab_prev_responses;
        e = lab_prev_errors; us = lab_prev_us;
    }
    snprintf(payload, sizeof(payload), "%s|%s|%s|%lld|%lld|%llu|%llu|%llu|%llu",
            ip, seq, nonce, (long long)lab_begin, (long long)now, n, r, e, us);
    ogs_hmac_sha256(key, 32, (uint8_t *)payload, strlen(payload), digest, 32);
    lab_hex(digest, mac);
    snprintf(body, sizeof(body), "{\"payload\":\"%s\",\"mac\":\"%s\"}", payload, mac);
    lab_prev_requests = lab_requests; lab_prev_responses = lab_responses;
    lab_prev_errors = lab_errors; lab_prev_us = lab_service_us;
    lab_requests = lab_responses = lab_errors = lab_service_us = 0;
    lab_begin = now;
    response = ogs_sbi_response_new();
    response->status = 200;
    response->http.content = ogs_strdup(body);
    response->http.content_length = strlen(body);
    ogs_sbi_header_set(response->http.headers, "content-type", "application/json");
    server_send_response(stream, response);
    return 1;
forbidden:
    response = ogs_sbi_response_new();
    response->status = 403;
    server_send_response(stream, response);
    return 1;
}

static void lab_record_response(ogs_sbi_stream_t *stream, int status)
{
    if (!getenv("LAB_KIE_KEY") || !stream->lab_started) return;
    lab_responses++;
    if (status >= 400) lab_errors++;
    lab_service_us += (unsigned long long)(ogs_time_now() - stream->lab_started);
    stream->lab_started = 0;
}
