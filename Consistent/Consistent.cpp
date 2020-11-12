#include <iostream>

int main(int argc, char* argv[])
{
	int array_size = atoi(argv[1]);
	
	int *first_vector, *second_vector;
	first_vector = (int*)calloc(array_size, sizeof(int));
	second_vector = (int*)calloc(array_size, sizeof(int));
	int i;
	for (i = 0; i < array_size; ++i)
	{
		first_vector[i] = 1;
		second_vector[i] = 1;
	}

	int* result_vector;
	result_vector = (int*)calloc(array_size, sizeof(int));

	for (i = 0; i < array_size; ++i)
	{
		result_vector[i] = first_vector[i] + second_vector[i];
	}


	printf("\nFirst item of result = %d", result_vector[0]);

	return 0;
}
