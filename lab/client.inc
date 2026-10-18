/* Trusted observer in the SCP process only. SPDX-License-Identifier: AGPL-3.0-or-later */
#include <unistd.h>
static void lab_observe(CURL *easy, const char *url, long status)
{
    const char *dir = getenv("LAB_OBSERVER_DIR");
    char exe[1024], path[1200];
    ssize_t n;
    double elapsed = 0;
    FILE *f;
    if (!dir || !url) return;
    n = readlink("/proc/self/exe", exe, sizeof(exe)-1);
    if (n < 0) return;
    exe[n] = 0;
    if (!strstr(exe, "open5gs-scpd")) return;
    curl_easy_getinfo(easy, CURLINFO_TOTAL_TIME, &elapsed);
    snprintf(path, sizeof(path), "%s/scp.tsv", dir);
    f = fopen(path, "a");
    if (!f) return;
    fprintf(f, "%lld\t%.3f\t%ld\t%s\n", (long long)ogs_time_now(),
            elapsed * 1000000.0, status, url);
    fclose(f);
}
