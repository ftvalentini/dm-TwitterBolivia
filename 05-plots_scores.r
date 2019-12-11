library(tidyverse)
library(magrittr)

# read --------------------------------------------------------------------

dat_0 = readr::read_delim("data/working/time_tokens_rel_0.csv", delim=",") %>%
  janitor::clean_names() 

dat_5000 = readr::read_delim("data/working/time_tokens_rel_5000.csv", delim=",") %>%
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

# funcion
save_frec_plots = function(tabla) {
  terminos = tabla %>% select(-c(created, clase)) %>% names()
  p_list = map(terminos, function(t) frec_plot(tabla, t)) %>% setNames(terminos)
  for (t in terminos) ggsave(paste0("output/plots/term_",t,".png"), p_list[[t]]
                             ,width=7, height=5)
  
}

# run
save_frec_plots(dat_0)
save_frec_plots(dat_5000)
