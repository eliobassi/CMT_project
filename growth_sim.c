#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAXLINE 512
#define STEPS_PER_YEAR 100
#define DT (1.0 / STEPS_PER_YEAR)

double update_biomass(double B, double P, double r0, double alpha, double K) {
    double r = r0 * exp(-alpha * P);
    double dBdt = r * B * (1.0 - B / K);
    return B + DT * dBdt;
}

int main() {
    FILE *fp = fopen("c_input.csv", "r");
    if (!fp) {
        printf("Error: cannot open c_input.csv\n");
        return 1;
    }

    char line[MAXLINE];

    // Read header
    fgets(line, MAXLINE, fp);

    printf("Region,Year,B_predicted\n");

    while (fgets(line, MAXLINE, fp)) {
        char region[50];
        int year;
        double B_obs, P, r_est, K, B0, r0, alpha;

        sscanf(line, "%[^,],%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
            region, &year, &B_obs, &P, &r_est, &K, &B0, &r0, &alpha);

        // Initialize biomass at B0 from the fitted model
        double B = B0;

        // Simulate one year with 100 small steps
        for (int step = 0; step < STEPS_PER_YEAR; step++) {
            B = update_biomass(B, P, r0, alpha, K);
        }

        printf("%s,%d,%.6f\n", region, year, B);
    }

    fclose(fp);
    return 0;
}

