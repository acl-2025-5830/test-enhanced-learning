# Save file or directory format:

## Default Format:
`records/{reproduct target}/{training domain}_{evaluation domain}_{baseline model}-{training method}`

The `training domain` is "base" if the results are from the base model.

## For Main Figure 4 (Long Term Retention):
`records/{reproduct target}/{training domain}_{overwrite domain}_{evaluation domain}_{baseline model}-{training method}`


We have removed a few configuration items from the result files (specifically `model_args`, `model_name`, and `model_name_sanitized`) to ensure anonymity.