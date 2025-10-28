pacman::p_load(
  Ipaper, data.table, dplyr, lubridate, 
  ggplot2
)

df = fread("./NoahPy_output.txt")
dat = select(df, Date, starts_with("SMC")) %>% 
  melt(c("Date"), variable.name = "Layer", value.name = "SMC")

p <- ggplot(dat, aes(Date, SMC)) + 
  geom_line(aes(color = Layer)) +
  facet_wrap(~Layer) +
  theme(legend.position = "bottom")

write_fig(p, 'd:/Rplot.pdf', 10, 8)
