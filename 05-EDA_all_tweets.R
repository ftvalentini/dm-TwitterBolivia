
library(tidyverse)
library(data.table)
library(forcats)
library(hrbrthemes)
library(ggtech)

p = "D://Data Science/Maestria UBA/github/dm-TwitterBolivia/"

all_tweets = readr::read_delim(paste0(p,"data/working/all_tweets.csv"), delim=",") %>%
  janitor::clean_names() 

all_tweets = all_tweets %>%
  select(-x1)

# Summary ----

tweets_por_clase = all_tweets %>%
  group_by(clase) %>%
  summarise(tuits = n())


ids_por_clase = all_tweets %>%
  group_by(clase) %>%
  summarise(count = n_distinct(user_id))


# Antiguedad users ----

  # Tabla

dist_tiempo_user = all_tweets %>%
  group_by(clase) %>%
  group_modify(~ {
    quantile(.x$tiempo_user, probs = c(0.25, 0.5, 0.75)) %>%
      tibble::enframe(name = "prob", value = "quantile")
  })


  # Plot

p0 = ggplot(all_tweets, aes( x = fct_relevel(clase,"AE","PE","-"), y = tiempo_user)) +
  geom_boxplot()
# compute lower and upper whiskers
ylim1 = boxplot.stats(all_tweets$tiempo_user)$stats[c(1, 5)]

# scale y limits based on ylim1
(p1 = p0 + coord_cartesian(ylim = ylim1*1.05) +
  ggtitle("Antigüedad de las cuentas de Twitter") + 
  scale_x_discrete(labels=c("Anti Evo","Pro Evo","Indefinido")) + 
  ylab("Dias de antigüedad") + 
  xlab("") + 
    theme_minimal(base_size = 15))

## Export

  