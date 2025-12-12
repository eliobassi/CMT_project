#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_LINE 256

// Logistic update: B(t+1) = B(t) + r(P)*B(t)*(1 - B(t)/K)
double logistic_step(double B, double r, double K) {
    return B + r * B * (1.0 - B / K);
}

// Compute r(P) = r_est * exp(-P)
double r_of_P(double r_est, double P) {
    return r_est * exp(-P);
}

// Process one scenario CSV
void compute_ndvi_scenario(const char* input_csv, const char* output_csv, double B0_initial) {
    FILE *fin = fopen(input_csv, "r");
    if (!fin) {
        printf("Error: cannot open %s\n", input_csv);
        return;
    }

    FILE *fout = fopen(output_csv, "w");
    if (!fout) {
        printf("Error: cannot write %s\n", output_csv);
        fclose(fin);
        return;
    }

    char line[MAX_LINE];
    int year;
    double P, K, B0, r_est;

    // Write header
    fprintf(fout, "Year,P,NDVI,K_estimated,B0_estimated,r_estimated\n");

    // Skip header of input CSV
    fgets(line, MAX_LINE, fin);

    double B = B0_initial;

    while (fgets(line, MAX_LINE, fin)) {
        sscanf(line, "%d,%lf,%lf,%lf,%lf", &year, &P, &K, &B0, &r_est);

        double rP = r_of_P(r_est, P);
        B = logistic_step(B, rP, K);

        if (B < 0) B = 0;
        if (B > 1) B = 1;

        fprintf(fout, "%d,%.10f,%.10f,%.10f,%.10f,%.10f\n",
                year, P, B, K, B0, r_est);
    }

    fclose(fin);
    fclose(fout);

    printf("✅ Scenario processed: %s → %s\n", input_csv, output_csv);
}

int main() {
    // Initial NDVI = NDVI 2018 (à mettre à jour selon ton fichier)
    double B0_initial = 0.5752443361584014;

    compute_ndvi_scenario("scenario_pollution_up.csv", 
                          "scenario_ndvi_up.csv", 
                          B0_initial);

    compute_ndvi_scenario("scenario_pollution_constant.csv", 
                          "scenario_ndvi_constant.csv", 
                          B0_initial);

    compute_ndvi_scenario("scenario_pollution_down.csv", 
                          "scenario_ndvi_down.csv", 
                          B0_initial);

    return 0;
}
