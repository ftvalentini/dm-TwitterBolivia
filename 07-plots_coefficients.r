library(tidyverse)
library(magrittr)

# read --------------------------------------------------------------------

# tfidf
dat_a = readr::read_delim("data/working/weights_tfidf.csv", delim=","
                        ,col_names=c("variable","weight")) %>%
  janitor::clean_names() 
# tfidf + user_features
dat_b = readr::read_delim("data/working/weights_tfidf_featusers.csv", delim=","
                        ,col_names=c("variable","weight")) %>%
  janitor::clean_names() 


# plot function -----------------------------------------------------------

coef_plot = function(tabla, model_title) {
  gdat = tabla %>% 
    arrange(-abs(weight)) %>% 
    head(20) %>% 
    mutate(clase = case_when(
      str_detect(variable,"^abt_") ~ "u_feat"
      ,weight>0 ~ "PE"
      ,TRUE ~ "AE"
    ))
  ggplot(gdat, aes(x=reorder(variable,-abs(weight)), y=weight)) +
    geom_col(aes(fill=clase)) +
    theme_minimal() + 
    theme(axis.text.x = element_text(angle=60, size=11)) +
    labs(
      title = model_title
      ,subtitle="Coeficientes de mayor valor absoluto"
      ,x=NULL, y="Coef."
    ) +
    NULL
}

# save plots -----------------------------------------------------------

g_a = coef_plot(dat_a, "Logistica tf-idf")
g_b = coef_plot(dat_b, "Logistica tf-idf + user_features")

ggsave("output/plots/weights_tfidf.png", g_a, width=8, height=6)
ggsave("output/plots/weights_tfidf_ufeat.png", g_b, width=8, height=6)
