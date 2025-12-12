
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* -----------------------------------------------------------
   Simulation NDVI (B) par modèle logistique à pas annuel :
   r(P) = r0 * exp(P)
   dB/dt = r(P) * B * (1 - B/K)
   ----------------------------------------------------------- */

#define MAX_ROWS 512
#define MAX_LINE 2048

typedef struct {
    int   year;
    double P;
    double r0;
    double K;
    double B0;
} Row;

typedef struct {
    int n;
    Row rows[MAX_ROWS];
} Table;

/* Trim utilitaire (enlève espaces et \n en bout de chaîne) */
static void rtrim(char *s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n-1] == ' ' || s[n-1] == '\t' || s[n-1] == '\r' || s[n-1] == '\n')) {
        s[--n] = '\0';
    }
}

/* Trouve l'indice d'une colonne dans l'en-tête CSV */
static int find_col(char **cols, int ncols, const char *name) {
    for (int i = 0; i < ncols; ++i) {
        if (strcmp(cols[i], name) == 0) return i;
    }
    return -1;
}

/* Lecture d'un CSV : on recherche Year, P, r0, K, B0 (ordre quelconque) */
static int read_csv(const char *path, Table *t) {
    t->n = 0;
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Erreur: impossible d'ouvrir %s\n", path);
        return 0;
    }

    char line[MAX_LINE];

    // Lire header
    if (!fgets(line, sizeof(line), f)) {
        fprintf(stderr, "Erreur: fichier vide %s\n", path);
        fclose(f);
        return 0;
    }
    rtrim(line);

    // Découper l'en-tête
    char *cols[64];
    int ncols = 0;
    {
        char *p = strtok(line, ",");
        while (p && ncols < 64) {
            rtrim(p);
            cols[ncols++] = p;
            p = strtok(NULL, ",");
        }
    }

    int idxYear = find_col(cols, ncols, "Year");
    int idxP    = find_col(cols, ncols, "P");
    int idxr0   = find_col(cols, ncols, "r0");
    int idxK    = find_col(cols, ncols, "K");
    int idxB0   = find_col(cols, ncols, "B0");

    if (idxYear<0 || idxP<0 || idxr0<0 || idxK<0 || idxB0<0) {
        fprintf(stderr, "Erreur: colonnes attendues manquantes dans %s\n", path);
        fclose(f);
        return 0;
    }

    // Lire les lignes
    while (fgets(line, sizeof(line), f)) {
        rtrim(line);
        if (line[0] == '\0') continue;

        // On duplique la ligne pour strtok (qui modifie la chaîne)
        char buf[MAX_LINE];
        strncpy(buf, line, sizeof(buf));
        buf[sizeof(buf)-1] = '\0';

        char *vals[64];
        int n = 0;
        char *p = strtok(buf, ",");
        while (p && n < 64) {
            vals[n++] = p;
            p = strtok(NULL, ",");
        }
        if (ncols > n) {
            fprintf(stderr, "Avertissement: ligne incomplète ignorée: %s\n", line);
            continue;
        }

        if (t->n >= MAX_ROWS) {
            fprintf(stderr, "Erreur: dépassement MAX_ROWS\n");
            fclose(f);
            return 0;
        }

        Row r;
        r.year = (int)strtol(vals[idxYear], NULL, 10);
        r.P  = strtod(vals[idxP],  NULL);
        r.r0 = strtod(vals[idxr0], NULL);
        r.K  = strtod(vals[idxK],  NULL);
        r.B0 = strtod(vals[idxB0], NULL);

        t->rows[t->n++] = r;
    }

    fclose(f);

    // Tri par année (simple insertion sort, n petit)
    for (int i = 1; i < t->n; ++i) {
        Row key = t->rows[i];
        int j = i - 1;
        while (j >= 0 && t->rows[j].year > key.year) {
            t->rows[j+1] = t->rows[j];
            --j;
        }
        t->rows[j+1] = key;
    }

    return 1;
}

/* Étape logistique analytique sur 1 an, r(P) = r0 * exp(P) */
static double logistic_step(double B, double r0, double P, double K) {
    const double eps = 1e-12;
    if (K <= eps) return fmin(1.0, fmax(0.0, B)); // sécurité
    double r = r0 * exp(P);
    // éviter division par zéro si B ~ 0
    double Bsafe = (B < eps) ? eps : B;
    double factor = exp(-r * 1.0); // Δt = 1 an
    double nextB = K / (1.0 + (K / Bsafe - 1.0) * factor);
    // clamp NDVI dans [0,1]
    if (nextB < 0.0) nextB = 0.0;
    if (nextB > 1.0) nextB = 1.0;
    return nextB;
}

/* Simule NDVI année par année pour un tableau de paramètres */
static void simulate_series(const Table *tab, double *ndvi_out) {
    if (tab->n <= 0) return;
    // B initial = B0 de la première ligne (ex. 2019)
    double B = tab->rows[0].B0;
    if (B < 0.0) B = 0.0;
    if (B > 1.0) B = 1.0;

    for (int i = 0; i < tab->n; ++i) {
        const Row *r = &tab->rows[i];
        // un pas de temps d'un an avec P, r0, K de l'année i
        B = logistic_step(B, r->r0, r->P, r->K);
        ndvi_out[i] = B;
    }
}

/* Cherche l'indice d'une année donnée dans un tableau trié par années */
static int index_of_year(const Table *t, int year) {
    // t->rows est trié ; recherche linéaire (n petit)
    for (int i = 0; i < t->n; ++i)
        if (t->rows[i].year == year) return i;
    return -1;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr,
            "Usage: %s <file_up> <file_down> <file_cst> <output_csv>\n"
            "  file_up   : scenario P +1%%/an (ex: scenario_P_up.csv)\n"
            "  file_down : scenario P -1%%/an (ex: scenario_P_down.csv)\n"
            "  file_cst  : scenario P constant (ex: scenario_P_constant.csv)\n"
            "  output_csv: fichier de sortie fusionné\n", argv[0]);
        return 1;
    }

    const char *file_up   = argv[1];
    const char *file_down = argv[2];
    const char *file_cst  = argv[3];
    const char *out_path  = argv[4];

    Table T_up, T_down, T_cst;

    if (!read_csv(file_up, &T_up))   return 1;
    if (!read_csv(file_down, &T_down)) return 1;
    if (!read_csv(file_cst, &T_cst))   return 1;

    // Simuler NDVI pour chaque scénario
    double *NDVI_up   = (double*)calloc(T_up.n,   sizeof(double));
    double *NDVI_down = (double*)calloc(T_down.n, sizeof(double));
    double *NDVI_cst  = (double*)calloc(T_cst.n,  sizeof(double));
    if (!NDVI_up || !NDVI_down || !NDVI_cst) {
        fprintf(stderr, "Erreur: allocation mémoire\n");
        free(NDVI_up); free(NDVI_down); free(NDVI_cst);
        return 1;
    }

    simulate_series(&T_up,   NDVI_up);
    simulate_series(&T_down, NDVI_down);
    simulate_series(&T_cst,  NDVI_cst);

    // Écrire la fusion : intersection des années présentes dans les 3 fichiers
    FILE *out = fopen(out_path, "w");
    if (!out) {
        fprintf(stderr, "Erreur: impossible d'ouvrir sortie %s\n", out_path);
        free(NDVI_up); free(NDVI_down); free(NDVI_cst);
        return 1;
    }

    fprintf(out, "Year,P_up,NDVI_up,P_down,NDVI_down,P_cst,NDVI_cst\n");

    // On parcourt les années de l'un des scénarios (ex: up), et on écrit seulement si l'année est dans les 3
    for (int i = 0; i < T_up.n; ++i) {
        int year = T_up.rows[i].year;
        int j = index_of_year(&T_down, year);
        int k = index_of_year(&T_cst,  year);
        if (j < 0 || k < 0) continue;

        double P_up_val   = T_up.rows[i].P;
        double P_down_val = T_down.rows[j].P;
        double P_cst_val  = T_cst.rows[k].P;

        double ndvi_up    = NDVI_up[i];
        double ndvi_down  = NDVI_down[j];
        double ndvi_cst   = NDVI_cst[k];

        // Écrire la ligne
        fprintf(out, "%d,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g\n",
                year, P_up_val, ndvi_up, P_down_val, ndvi_down, P_cst_val, ndvi_cst);
    }

    fclose(out);
    free(NDVI_up    free(NDVI_up); free(NDVI_down); free(NDVI_cst);

    return 0;
