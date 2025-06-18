#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>      // Para funciones MPI
#include <omp.h>      // Para directivas OpenMP
#include <sys/time.h> // Para medición de tiempo

// Definiciones de tamaño de matriz (ejemplo, se pueden pasar como argumentos)
#define MATRIX_SIZE 1024
#define A MATRIX_SIZE
#define B MATRIX_SIZE
#define C MATRIX_SIZE

// Función para inicializar una matriz con valores aleatorios
void initialize_matrix(int *mat, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            mat[i * cols + j] = rand() % 10; // Valores pequeños para ejemplo
        }
    }
}

// Función para imprimir una matriz (opcional, para depuración)
void print_matrix(int *mat, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    int rank, num_procs;
    double start_time, end_time;

    // 1. Inicialización MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Validar argumentos para número de hilos OpenMP
    if (argc != 2)
    {
        if (rank == 0)
        {
            printf("Uso: %s <num_hilos_openmp>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int num_threads_openmp = atoi(argv[1]);
    if (num_threads_openmp <= 0)
    {
        if (rank == 0)
        {
            printf("El número de hilos OpenMP debe ser positivo.\n");
        }
        MPI_Finalize();
        return 1;
    }
    omp_set_num_threads(num_threads_openmp); // Establece el número de hilos para OpenMP

    int *matrix_A = NULL;
    int *matrix_B = NULL;
    int *matrix_C = NULL;

    int *local_matrix_A_rows = NULL; // Filas de A para cada proceso
    int *local_matrix_C_rows = NULL; // Filas de C resultantes para cada proceso

    int rows_per_proc = A / num_procs;
    int remainder_rows = A % num_procs;
    int current_proc_rows = rows_per_proc;
    if (rank < remainder_rows)
    {
        current_proc_rows++;
    }

    // Proceso raíz (rank 0) se encarga de la inicialización y recolección
    if (rank == 0)
    {
        matrix_A = (int *)malloc(A * B * sizeof(int));
        matrix_B = (int *)malloc(B * C * sizeof(int));
        matrix_C = (int *)malloc(A * C * sizeof(int));

        if (matrix_A == NULL || matrix_B == NULL || matrix_C == NULL)
        {
            fprintf(stderr, "Error: No se pudo asignar memoria para las matrices en el proceso raíz.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        initialize_matrix(matrix_A, A, B);
        initialize_matrix(matrix_B, B, C);
        // print_matrix(matrix_A, A, B); // Para depuración
        // print_matrix(matrix_B, B, C); // Para depuración

        printf("Iniciando multiplicación de matrices con %d procesos MPI y %d hilos OpenMP por proceso...\n", num_procs, num_threads_openmp);
        start_time = MPI_Wtime(); // Inicio de medición de tiempo total
    }

    // Broadcast de la matriz B a todos los procesos
    matrix_B = (int *)malloc(B * C * sizeof(int));
    if (matrix_B == NULL)
    {
        fprintf(stderr, "Error: No se pudo asignar memoria para matrix_B en el proceso %d.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Bcast(matrix_B, B * C, MPI_INT, 0, MPI_COMM_WORLD);

    // Determinar los sendcounts y displs para MPI_Scatterv
    int *sendcounts_A = NULL;
    int *displs_A = NULL;
    int *sendcounts_C = NULL;
    int *displs_C = NULL;

    if (rank == 0)
    {
        sendcounts_A = (int *)malloc(num_procs * sizeof(int));
        displs_A = (int *)malloc(num_procs * sizeof(int));
        sendcounts_C = (int *)malloc(num_procs * sizeof(int));
        displs_C = (int *)malloc(num_procs * sizeof(int));

        int current_disp = 0;
        for (int i = 0; i < num_procs; i++)
        {
            int proc_rows = rows_per_proc;
            if (i < remainder_rows)
            {
                proc_rows++;
            }
            sendcounts_A[i] = proc_rows * B; // Número de elementos para matrix A
            displs_A[i] = current_disp;
            sendcounts_C[i] = proc_rows * C;    // Número de elementos para matrix C
            displs_C[i] = current_disp * C / B; // Desplazamiento para matrix C (asumiendo que se distribuyen filas completas)
            current_disp += proc_rows * B;
        }
    }

    local_matrix_A_rows = (int *)malloc(current_proc_rows * B * sizeof(int));
    local_matrix_C_rows = (int *)malloc(current_proc_rows * C * sizeof(int));

    if (local_matrix_A_rows == NULL || local_matrix_C_rows == NULL)
    {
        fprintf(stderr, "Error: No se pudo asignar memoria para las matrices locales en el proceso %d.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 2. Distribución de Datos (Filas de A a cada proceso)
    // MPI_Scatterv permite enviar porciones de tamaño variable
    MPI_Scatterv(matrix_A, sendcounts_A, displs_A, MPI_INT,
                 local_matrix_A_rows, current_proc_rows * B, MPI_INT,
                 0, MPI_COMM_WORLD);

// 3. Cálculo Paralelo (Multiplicación de Submatrices con OpenMP)
// Cada proceso calcula su parte de la matriz C
#pragma omp parallel for
    for (int i = 0; i < current_proc_rows; i++)
    {
        for (int j = 0; j < C; j++)
        {
            local_matrix_C_rows[i * C + j] = 0; // Inicializar a cero
            for (int k = 0; k < B; k++)
            {
                local_matrix_C_rows[i * C + j] += local_matrix_A_rows[i * B + k] * matrix_B[k * C + j];
            }
        }
    }

    // 4. Recolección de Resultados
    // MPI_Gatherv permite recibir porciones de tamaño variable
    MPI_Gatherv(local_matrix_C_rows, current_proc_rows * C, MPI_INT,
                matrix_C, sendcounts_C, displs_C, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        end_time = MPI_Wtime(); // Fin de medición de tiempo total
        double elapsed_time = end_time - start_time;
        printf("Tiempo de ejecución total: %.6f segundos\n", elapsed_time);

        // Opcional: Verificar la suma de elementos para validación
        long long total_sum_C = 0;
        for (int i = 0; i < A * C; i++)
        {
            total_sum_C += matrix_C[i];
        }
        printf("Suma total de elementos en la matriz C: %lld\n", total_sum_C);

        // Liberar memoria en el proceso raíz
        free(matrix_A);
        free(matrix_B);
        free(matrix_C);

        // Liberar sendcounts y displs
        free(sendcounts_A);
        free(displs_A);
        free(sendcounts_C);
        free(displs_C);
    }

    // Liberar memoria local en todos los procesos
    free(local_matrix_A_rows);
    free(local_matrix_C_rows);
    if (rank != 0)
    { // Si matrix_B no fue malloc'd en rank 0
        free(matrix_B);
    }

    // Finalización MPI
    MPI_Finalize();

    return 0;
}