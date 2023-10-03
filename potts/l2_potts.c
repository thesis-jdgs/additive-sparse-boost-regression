#include <stdlib.h>

/**
 * @brief Calculate piecewise constant approximations using Potts regularization.
 *
 * This function calculates piecewise constant approximations of data using
 * Potts regularization. It returns the number of leaves in the piecewise
 * approximation.
 *
 * @param input_data The input data array.
 * @param weights The array of weights associated with input_data.
 * @param data_length The length of input_data and weights arrays.
 * @param l0_fused_regularization The L0 fused regularization parameter.
 * @param l2_regularization The L2 regularization parameter.
 * @param excluded_interval_size The maximum interval size to exclude.
 * @param split_indexes Output array to store split indexes.
 * @param leaves Output array to store the piecewise constant values.
 * @return The number of leaves in the piecewise approximation.
 */
int l2_potts(
    double* input_data,
    double* weights,
    const int data_length,
    const double l0_fused_regularization,
    const double l2_regularization,
    const int excluded_interval_size,
    int* split_indexes,
    double* leaves
){
    double* cumulative_first_moments = malloc((data_length + 1)*sizeof(double));
    double* cumulative_second_moments = malloc((data_length + 1)*sizeof(double));
    double* cumulative_weights = malloc((data_length + 1)*sizeof(double));
    cumulative_first_moments[0] = 0.0;
    cumulative_second_moments[0] = 0.0;
    cumulative_weights[0] = 0.0;

    double weight, weighted_data;
    for (int j = 0; j < data_length; j++){
        weight = weights[j];
        weighted_data = weight*input_data[j];
        cumulative_first_moments[j + 1] = cumulative_first_moments[j] + weighted_data;
        cumulative_second_moments[j + 1] = cumulative_second_moments[j] + weighted_data*input_data[j];
        cumulative_weights[j + 1] = cumulative_weights[j] + weight;
    }

    int* jumps = malloc(data_length*sizeof(int));
    double* potts_values = malloc(data_length*sizeof(double));
    double deviation, candidate, first_moments, first_moments_difference;
    for (int right = 1; right <= data_length; right++){
        first_moments = cumulative_first_moments[right];
        potts_values[right - 1] = cumulative_second_moments[right] -
        first_moments*first_moments/(cumulative_weights[right] + l2_regularization*right);
        jumps[right - 1] = 0;

        for (int left = right - excluded_interval_size - 1; left >= 1; left--){
            first_moments_difference = first_moments - cumulative_first_moments[left];
            deviation = l0_fused_regularization +
                cumulative_second_moments[right] - cumulative_second_moments[left] -
                first_moments_difference*first_moments_difference/
                (cumulative_weights[right] - cumulative_weights[left] +
                l2_regularization*(right- left));
            if (deviation > potts_values[right - 1]){
                break;
            }
            candidate = potts_values[left - 1] + deviation;
            if (candidate < potts_values[right - 1]){
                potts_values[right - 1] = candidate;
                jumps[right - 1] = left;
            }
        }
    }

    int right = data_length;
    int left = jumps[data_length - 1];
    double mean;
    int leave_count = 0;
    while (right > 0){
        mean = (cumulative_first_moments[right] - cumulative_first_moments[left])/
            (cumulative_weights[right] - cumulative_weights[left] + l2_regularization*(right- left));
        leaves[leave_count] = mean;
        split_indexes[leave_count] = left;
        right = left;
        if (right < 1){
            break;
        }
        left = jumps[right - 1];
        leave_count++;
    }

    free(cumulative_first_moments);
    free(cumulative_second_moments);
    free(cumulative_weights);
    free(jumps);
    free(potts_values);

    return leave_count;
}

#ifdef _WIN32
__declspec(dllexport) int l2_potts(
    double* input_data,
    double* weights,
    const int data_length,
    const double l0_fused_regularization,
    const double l2_regularization,
    const int excluded_interval_size,
    int* split_indexes,
    double* leaves
);
#endif
