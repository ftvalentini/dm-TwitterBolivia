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

coef_plot = function(tabla, rows_index, model_title) {
  gdat = tabla %>% 
    arrange(-abs(weight)) %>% 
    slice(rows_index) %>% 
    mutate(clase = case_when(
      str_detect(variable,"^abt_") ~ "u_feat"
      ,weight>0 ~ "PE"
      ,TRUE ~ "AE"
    ))
  cols = c("AE"="#F8766D", "PE"="#00BFC4", "u_feat"="#9C9A99")
  ggplot(gdat, aes(x=reorder(variable,-abs(weight)), y=weight)) +
    geom_col(aes(fill=clase)) +
    theme_minimal() + 
    scale_fill_manual(values=cols) +
    theme(axis.text.x = element_text(angle=60, size=12, hjust=1)) +
    labs(
      title = model_title
      ,subtitle=paste0("Coeficientes de mayor valor absoluto "
                       ,"(",rows_index[1],"-",rows_index[length(rows_index)],")")
      ,x=NULL, y="Coef."
    ) +
    NULL
}

# save plots -----------------------------------------------------------

g_a = coef_plot(dat_a, 1:30, "Logistica tf-idf")
g_b = coef_plot(dat_b, 1:30, "Logistica tf-idf + user_features")

ggsave("output/plots/weights_tfidf.png", g_a, width=8, height=4)
ggsave("output/plots/weights_tfidf_ufeat.png", g_b, width=8, height=4)
