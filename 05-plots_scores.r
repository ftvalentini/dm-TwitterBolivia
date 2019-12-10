library(tidyverse)
library(magrittr)

p = "MaestriaDM/Materias/Text_mining/tp/"

# dat = readr::read_delim(paste0(p,"data/working/time_tokens_abs_0.csv"), delim=",") %>%
  # janitor::clean_names() 
dat = readr::read_delim(paste0(p,"data/working/time_tokens_rel_5000.csv"), delim=",") %>%
  janitor::clean_names() 


# plot function -----------------------------------------------------------

frec_plot = function(tabla, termino_q) {
  ggplot(data=tabla, aes(x=created, y=!!rlang::sym(termino_q), color=clase)) +
    geom_point() +
    geom_smooth(method="loess",se=F) +
    theme_minimal() + 
    labs(
      title = termino_q
      # ,subtitle="% de tweets con t?rmino sobre total de la clase"
      ,x=NULL, y=NULL
    ) +
    NULL
}

# save plots -----------------------------------------------------------

# TODO

# names(dat)
# frec_plot(dat, "dictador")
# frec_plot(dat, "fraude")
# frec_plot(dat, "golpe")
# 
# frec_plot(dat, "saqueando_quemando_actos_vandalicos")
# frec_plot(dat, "militantes_atentando")
# 
# dat
