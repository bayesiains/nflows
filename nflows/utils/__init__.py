from nflows.utils.io import (
    get_checkpoint_root,
    get_data_root,
    get_log_root,
    get_output_root,
    get_project_root,
    get_timestamp,
    is_on_cluster,
)
from nflows.utils.mmd import biased_mmd, unbiased_mmd_squared
from nflows.utils.plot import plot_hist_marginals, plot_hist_marginals_pair
from nflows.utils.torchutils import (
    cbrt,
    create_alternating_binary_mask,
    create_mid_split_binary_mask,
    create_random_binary_mask,
    gaussian_kde_log_eval,
    get_num_parameters,
    get_temperature,
    logabsdet,
    merge_leading_dims,
    notinfnotnan,
    random_orthogonal,
    repeat_rows,
    searchsorted,
    split_leading_dim,
    sum_except_batch,
    tensor2numpy,
    tile,
)
from nflows.utils.typechecks import (
    is_bool,
    is_int,
    is_nonnegative_int,
    is_positive_int,
    is_power_of_two,
)

# this causes problems
# -> create nflows.models and a dynamic dispatch mechanism
# from nflows.utils.get_models import (
#     get_classifier,
#     get_neural_likelihood,
#     get_neural_posterior,
# )
