# Multiplicación de Matrices Híbrida (MPI + OpenMP)

Este repositorio contiene una implementación en C para la multiplicación de dos matrices grandes utilizando un enfoque de programación paralela híbrida. El objetivo es combinar las capacidades de **Message Passing Interface (MPI)** para la distribución de tareas entre múltiples procesos y **Open Multi-Processing (OpenMP)** para la paralelización a nivel de hilos dentro de cada proceso.
## Estructura y Funcionamiento del Script

El código `matrix_mult.c` está diseñado para realizar la multiplicación de dos matrices, `A` y `B`, generando una matriz `C` como resultado (`C = A x B`). Su funcionamiento se divide en varias etapas clave para implementar el paralelismo híbrido.

### Inicialización Global

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <sys/time.h>

#define MATRIX_SIZE 1024 // Define el tamaño de las matrices (MATRIX_SIZE x MATRIX_SIZE)

int main(int argc, char *argv[]) {
    int rank, num_procs;
    double start_time, end_time;

    MPI_Init(&argc, &argv); // Inicializa el entorno MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtiene el rango (ID) de este proceso
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); // Obtiene el número total de procesos MPI

    // Valida el argumento para el número de hilos OpenMP
    // ...
    int num_threads_openmp = atoi(argv[1]);
    omp_set_num_threads(num_threads_openmp); // Establece el número de hilos OpenMP para este proceso
    // ...
}
```

  * **Cabeceras:** Se incluyen las librerías estándar de C (`stdio.h`, `stdlib.h`), así como `mpi.h` para las funciones de MPI y `omp.h` para las directivas de OpenMP. `sys/time.h` se incluye para medir el tiempo, aunque se usa `MPI_Wtime` para una mayor precisión en entornos MPI.
  * **`MATRIX_SIZE`:** Una macro define la dimensión de las matrices, asumiendo matrices cuadradas de `1024x1024`.
  * **Inicialización MPI:** `MPI_Init` inicia el subsistema MPI. `MPI_Comm_rank` asigna un identificador único (rango) a cada instancia del programa, y `MPI_Comm_size` obtiene el número total de instancias lanzadas.
  * **Configuración OpenMP:** El programa requiere un argumento de línea de comandos (`argv[1]`) que especifica el número de hilos OpenMP. `omp_set_num_threads()` configura OpenMP para que las regiones paralelas posteriores dentro de este proceso MPI utilicen el número de hilos especificado.

### Preparación de Matrices y Distribución de Carga

```c
    int *matrix_A = NULL;
    int *matrix_B = NULL;
    int *matrix_C = NULL;

    int *local_matrix_A_rows = NULL; // Porción de filas de A para este proceso
    int *local_matrix_C_rows = NULL; // Porción de filas de C que este proceso calculará

    // Lógica para determinar cuántas filas de A y C corresponden a este proceso
    int rows_per_proc = MATRIX_SIZE / num_procs;
    int remainder_rows = MATRIX_SIZE % num_procs;
    int current_proc_rows = rows_per_proc;
    if (rank < remainder_rows) {
        current_proc_rows++; // Los primeros 'remainder_rows' procesos obtienen una fila extra
    }

    if (rank == 0) { // Solo el proceso raíz (rank 0) inicializa y asigna memoria para las matrices completas
        matrix_A = (int *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
        matrix_B = (int *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
        matrix_C = (int *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
        // ... inicializar matrix_A y matrix_B con valores aleatorios ...
        start_time = MPI_Wtime(); // Inicia la medición de tiempo total
    }

    // Todos los procesos asignan memoria para matrix_B (que será broadcasted) y sus porciones locales
    matrix_B = (int *)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(int)); // Asignación para recibir B
    local_matrix_A_rows = (int *)malloc(current_proc_rows * MATRIX_SIZE * sizeof(int));
    local_matrix_C_rows = (int *)malloc(current_proc_rows * MATRIX_SIZE * sizeof(int));
```

  * **Punteros a Matrices:** Se declaran punteros para las matrices `A`, `B`, `C` (globales) y para las porciones locales `local_matrix_A_rows` y `local_matrix_C_rows` que cada proceso MPI manejará.
  * **Balanceo de Carga:** El código calcula dinámicamente cuántas filas de la matriz `A` (y, por ende, de `C`) le corresponden a cada proceso MPI. Se distribuyen equitativamente las filas, y si hay un remanente, los primeros procesos reciben una fila adicional para asegurar una carga balanceada.
  * **Asignación de Memoria:**
      * El **proceso raíz (rank 0)** asigna memoria para las matrices completas `A`, `B` y `C`, e inicializa `A` y `B` con valores aleatorios. También es el punto donde comienza la medición del tiempo total con `MPI_Wtime()`.
      * **Todos los procesos (incluido el raíz)** asignan memoria para recibir la matriz `B` completa y para almacenar sus porciones locales de `A` y `C`.

### Distribución de `matrix_B` (Broadcast MPI)

```c
    MPI_Bcast(matrix_B, MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
```

  * `MPI_Bcast`: Esta es una operación de comunicación colectiva. El proceso raíz (`rank 0`) envía el contenido completo de `matrix_B` a todos los demás procesos en el comunicador `MPI_COMM_WORLD`. Esto es necesario porque cada proceso MPI necesita la matriz `B` completa para poder calcular los productos punto para sus filas asignadas de `C`.

### Distribución de `matrix_A` (Scatter MPI)

```c
    // Preparación de sendcounts y displs para MPI_Scatterv (solo en rank 0)
    int *sendcounts_A = NULL;
    int *displs_A = NULL;
    // ... lógica para calcular sendcounts_A y displs_A basados en current_proc_rows para cada proceso ...

    MPI_Scatterv(matrix_A, sendcounts_A, displs_A, MPI_INT,
                 local_matrix_A_rows, current_proc_rows * MATRIX_SIZE, MPI_INT,
                 0, MPI_COMM_WORLD);
```

  * **`sendcounts_A` y `displs_A`:** El proceso raíz calcula dos arrays auxiliares. `sendcounts_A` especifica cuántos elementos (enteros) de `matrix_A` se enviarán a cada proceso. `displs_A` especifica el desplazamiento (la posición inicial) en `matrix_A` desde donde el proceso raíz debe tomar los datos para enviar a cada proceso. Se usa `MPI_Scatterv` (la versión "variable" de `Scatter`) porque las porciones de filas asignadas pueden no ser de tamaño idéntico para todos los procesos si el número total de filas no es divisible entre el número de procesos.
  * `MPI_Scatterv`: Distribuye las filas de `matrix_A` desde el proceso raíz a los `local_matrix_A_rows` de cada proceso MPI. Cada proceso recibe solo la porción de `A` que le corresponde.

### Cálculo Paralelo Híbrido (OpenMP)

```c
    #pragma omp parallel for
    for (int i = 0; i < current_proc_rows; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            local_matrix_C_rows[i * MATRIX_SIZE + j] = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                local_matrix_C_rows[i * MATRIX_SIZE + j] += local_matrix_A_rows[i * MATRIX_SIZE + k] * matrix_B[k * MATRIX_SIZE + j];
            }
        }
    }
```

  * Esta es la parte central del cálculo. Cada proceso MPI, trabajando con su `local_matrix_A_rows` y la `matrix_B` completa, calcula su porción de la matriz `C` (`local_matrix_C_rows`).
  * `#pragma omp parallel for`: Esta directiva OpenMP es la clave del paralelismo a nivel de hilos. Indica al compilador que el bucle `for` inmediatamente siguiente (el que itera sobre `i`, las filas locales) debe ser distribuido y ejecutado en paralelo por los hilos OpenMP configurados en ese *mismo proceso MPI*. Cada hilo trabaja en un subconjunto de las `current_proc_rows`, realizando una parte de la multiplicación.

### Recolección de Resultados (Gather MPI)

```c
    // Preparación de sendcounts_C y displs_C para MPI_Gatherv (solo en rank 0)
    int *sendcounts_C = NULL;
    int *displs_C = NULL;
    // ... lógica para calcular sendcounts_C y displs_C similar a A ...

    MPI_Gatherv(local_matrix_C_rows, current_proc_rows * MATRIX_SIZE, MPI_INT,
                matrix_C, sendcounts_C, displs_C, MPI_INT,
                0, MPI_COMM_WORLD);
```

  * **`sendcounts_C` y `displs_C`:** El proceso raíz también calcula arrays similares para la operación de recolección. `sendcounts_C` indica cuántos elementos esperará el raíz de cada proceso, y `displs_C` indica dónde colocar esos elementos en la `matrix_C` final.
  * `MPI_Gatherv`: Esta operación colectiva se encarga de recopilar las porciones `local_matrix_C_rows` calculadas por cada proceso MPI y ensamblarlas en la `matrix_C` completa en el proceso raíz (`rank 0`).

### Medición de Tiempo y Finalización

```c
    if (rank == 0) {
        end_time = MPI_Wtime(); // Fin de la medición de tiempo total
        double elapsed_time = end_time - start_time;
        printf("Tiempo de ejecución total: %.6f segundos\n", elapsed_time);
        // ... liberación de memoria para las matrices completas y sendcounts/displs ...
    }

    // Liberación de memoria para las porciones locales en todos los procesos
    // ...
    MPI_Finalize(); // Finaliza el entorno MPI
}
```

  * `MPI_Wtime()`: El proceso raíz mide el tiempo transcurrido desde el inicio de la operación paralela hasta la recolección final, proporcionando el tiempo total de ejecución.
  * **Liberación de Memoria:** Es crucial que toda la memoria asignada dinámicamente (`malloc`) sea liberada (`free`) por el proceso correspondiente para evitar fugas de memoria.
  * `MPI_Finalize()`: Esta es la última llamada a la biblioteca MPI y limpia todos los recursos del entorno MPI.

## Requisitos

Para compilar y ejecutar este script, necesitarás:

  * Un compilador de C (como GCC).
  * Una implementación de MPI (como OpenMPI o MPICH).
  * Soporte para OpenMP (generalmente incluido con GCC).

## Cómo Compilar

Navega al directorio donde guardaste `matrix_mult.c` y ejecuta el siguiente comando:

```bash
mpicc -fopenmp matrix_mult.c -o matrix_mult
```

## 4\. Cómo Ejecutar

Una vez compilado, puedes ejecutar el programa especificando el número de procesos MPI (`-np`) y el número de hilos OpenMP por proceso como argumento:

```bash
mpiexec -np <numero_de_procesos_mpi> ./matrix_mult <numero_de_hilos_openmp>
```

**Ejemplos:**

  * **Para ejecutar con 1 proceso MPI y 1 hilo OpenMP (configuración secuencial para referencia):**
    ```bash
    mpiexec -np 1 ./matrix_mult 1
    ```
  * **Para ejecutar con 2 procesos MPI y 4 hilos OpenMP por proceso:**
    ```bash
    mpiexec -np 2 ./matrix_mult 4
    ```
  * **Para ejecutar con 4 procesos MPI y 2 hilos OpenMP por proceso:**
    ```bash
    mpiexec -np 4 ./matrix_mult 2
    ```
