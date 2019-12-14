library(tidyverse)
library(magrittr)

# read --------------------------------------------------------------------

dats_0 = list(
  "abs" = readr::read_delim("data/working/time_tokens_abs_0.csv", delim=",") %>% janitor::clean_names() 
  ,"rel" = readr::read_delim("data/working/time_tokens_rel_0.csv", delim=",") %>% janitor::clean_names() 
)

dats_5000 = list(
  "abs" = readr::read_delim("data/working/time_tokens_abs_5000.csv", delim=",") %>% janitor::clean_names() 
  ,"rel" = readr::read_delim("data/working/time_tokens_rel_5000.csv", delim=",") %>% janitor::clean_names() 
)

dats_3000 = list(
  "abs" = readr::read_delim("data/working/time_tokens_abs_3000.csv", delim=",") %>% janitor::clean_names() 
  ,"rel" = readr::read_delim("data/working/time_tokens_rel_3000.csv", delim=",") %>% janitor::clean_names() 
)


# plot function -----------------------------------------------------------

frec_plot = function(tablas, termino_q) {
  gdat = tablas$rel %>% select(created, clase, termino_q) %>% 
    inner_join(tablas$abs %>% select(created, clase, termino_q)
               , by=c("created","clase"), suffix = c("","_abs"))
  ggplot(data=gdat, aes(x=created, y=!!rlang::sym(termino_q), color=clase)) +
    geom_point(aes(alpha=!!rlang::sym(paste0(termino_q,"_abs")))) +
    geom_smooth(method="loess",se=F) +
    theme_minimal() + 
    labs(
      title = termino_q
      ,subtitle="% de tweets con término sobre total de la clase"
      ,x=NULL, y=NULL
      ,alpha="f. abs."
    ) +
    theme(
      legend.title=element_text(size=6)
      ,legend.text=element_text(size=5)
      ,plot.subtitle=element_text(size=9)
    ) +
    NULL
  }

# save plots -----------------------------------------------------------

# funcion
save_frec_plots = function(tablas) {
  terminos = tablas$rel %>% select(-c(created, clase)) %>% names()
  p_list = map(terminos, function(t) frec_plot(tablas, t)) %>% setNames(terminos)
  for (t in terminos) ggsave(paste0("output/plots/term_",t,".png"), p_list[[t]]
                             ,width=4, height=2.5)
  
}

# run
save_frec_plots(dats_0)
save_frec_plots(dats_3000)
save_frec_plots(dats_5000)




