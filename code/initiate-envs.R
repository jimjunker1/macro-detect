library(here)
library(knitr)
i_am("code/initiate-envs.R")
#setup the virtual environment for python
if (!dir.exists(here("virtualenv/"))) {
  reticulate::conda_create(here("virtualenv/"))
}
# the .Rprofile should automatically use this python version
reticulate::use_condaenv(here('virtualenv/'))
# let's install the required python modules
if(!all(reticulate::py_module_available('sklearn'),
        reticulate::py_module_available('PIL'),
        reticulate::py_module_available('matplotlib'),
        reticulate::py_module_available('json'))
){
  reticulate::conda_install(
    packages = c('scikit-learn', 'Pillow','matplotlib','json'),
    envname = here('virtualenv/')
  )}

if(!reticulate::py_module_available('tensorflow')){
  reticulate::py_install(
    packages = c('tensorflow'),
    envname = here('virtualenv/'),
    pip = TRUE
# you may run into issues with previous versions of tensorflow if installed before
# you can overwrite previous installations by uncommenting the line below
    #, pip_ignore_installed =TRUE
  )}
