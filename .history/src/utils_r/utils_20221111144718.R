library(CDM)
library(GDINA)

resp25_15 = read.csv('data/real/timss07/25_15/resp.csv')[-1]
q_25_15 = read.csv('data/real/timss07/25_15/q.csv')[-1]
q_25_7 = read.csv('data/real/timss07/25_7/q.csv')[-1]

lcdm = gdina(resp25_15, q.matrix=q_25_15, link="logit")
lcdm = gdina(resp25_15, q.matrix=q_25_7, link="logit")
anova(lcdm, lcdm)
AIC(lcdm)
