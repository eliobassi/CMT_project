#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAXLINE 512
#define STEPS_PER_YEAR 100
#define DT (1.0 / STEPS_PER_YEAR)
#define MAX_YEARS 100
#define MAX_REGIONS 300

// Mise à jour du NDVI
double update_biomass(double B, double P, double r0, double alpha, double K) {
    double r = r0 * exp(-alpha * P);
    double dBdt = r * B * (1.0 - B / K);
    return B + DT * dBdt;
}

// Lecture + Simulation par région
void simulate_growth(char *scenario_csv, char *output_csv) {
    FILE *fp = fopen(scenario_csv, "r");
    if (!fp) {
        printf("Erreur : impossible d’ouvrir %s\n", scenario_csv);
        return;
    }

    FILE *fout = fopen(output_csv, "w");
    if (!fout) {
        printf("Erreur : impossible d’ouvrir %s\n", output_csv);
        fclose(fp);
        return;
    }

    fprintf(fout, "Region,Year,B_predicted\n");

    char line[MAXLINE];

    // Lire l'en-tête
    fgets(line, MAXLINE, fp);

    char current_region[50] = "";
    double P_values[MAX_YEARS];
    int years[MAX_YEARS];
    int count = 0;

    double B0=0, r0=0, alpha=0, K=0;

    while (fgets(line, MAXLINE, fp)) {

        char region[50];
        int year;
        double P, r_est;

        // LECTURE CORRECTE DES COLONNES :
        // Region,Year,NO2,r_estimated,K_estimated,B0_estimated,r0_global,alpha_global
        sscanf(line, "%[^,],%d,%lf,%lf,%lf,%lf,%lf,%lf",
               region, &year, &P, &r_est, &K, &B0, &r0, &alpha);

        // Nouvelle région rencontrée → simuler la précédente
        if (strcmp(current_region, region) != 0 && count > 0) {

            double B = B0;

            for (int y = 0; y < count; y++) {
                for (int step = 0; step < STEPS_PER_YEAR; step++) {
                    B = update_biomass(B, P_values[y], r0, alpha, K);
                }
                fprintf(fout, "%s,%d,%.6f\n", current_region, years[y], B);
            }

            count = 0;
        }

        strcpy(current_region, region);

        P_values[count] = P;
        years[count] = year;
        count++;
    }

    // Simuler la dernière région
    if (count > 0) {
        double B = B0;

        for (int y = 0; y < count; y++) {
            for (int step = 0; step < STEPS_PER_YEAR; step++) {
                B = update_biomass(B, P_values[y], r0, alpha, K);
            }
            fprintf(fout, "%s,%d,%.6f\n", current_region, years[y], B);
        }
    }

    fclose(fp);
    fclose(fout);

    printf("Simulation terminée pour : %s → %s\n", scenario_csv, output_csv);
}

int main(int argc, char *argv[]) {

    simulate_growth("scenario_with_params_constant_clean.csv",
                    "NDVI_scenario_constant.csv");

    simulate_growth("scenario_with_params_minus1percent_clean.csv",
                    "NDVI_scenario_minus1percent.csv");

    simulate_growth("scenario_with_params_plus1percent_clean.csv",
                    "NDVI_scenario_plus1percent.csv");

    return 0;
}
