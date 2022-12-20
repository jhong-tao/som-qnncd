library(CDM)

q = read.csv("data/real/timss07/25_15/q.csv")[-1]
resp = read.csv("data/real/timss07/25_15/resp.csv")[-1]


progress = FALSE
maxit = 1000
DINA = gdina(resp, q.matrix=q, rule='DINA', maxit=maxit, progress=progress)
print(sprintf("DINA      Npars:%s     AIC:%s      BIC:%s", DINA$Npars, DINA$AIC, DINA$BIC))

rRUM = gdina(resp, q.matrix=q, rule='RRUM', maxit=maxit, progress=progress)
print(sprintf("rRUM      Npars:%s     AIC:%s      BIC:%s", rRUM$Npars, rRUM$AIC, rRUM$BIC))

ACDM = gdina(resp, q.matrix=q, rule='ACDM', maxit=maxit, progress=progress)
print(sprintf("ACDM      Npars:%s     AIC:%s      BIC:%s", ACDM$Npars, ACDM$AIC, ACDM$BIC))

LCDM = gdina(resp, q.matrix=q, link="logit", maxit=maxit, progress=progress)
print(sprintf("LCDM      Npars:%s     AIC:%s      BIC:%s", LCDM$Npars, LCDM$AIC, LCDM$BIC))

GDINA = gdina(resp, q.matrix=q, rule='GDINA', maxit=maxit, progress=progress)
print(sprintf("GDINA     Npars:%s     AIC:%s      BIC:%s", GDINA$Npars, GDINA$AIC, GDINA$BIC))

q = read.csv("G4_py/698/q.csv")[-1]
resp = read.csv("G4_py/698/resp.csv")[-1]
LCDM = gdina(resp, q.matrix=q, link="logit", maxit=maxit, progress=progress)
target = IRT.factor.scores(LCDM)
write.csv(target, 'G4_py/698/label.csv')


q = read.csv("G4_lee/698/q.csv")[-1]
resp = read.csv("G4_lee/698/resp.csv")[-1]
rRUM = gdina(resp, q.matrix=q, rule='RRUM', maxit=maxit, progress=progress)
target = IRT.factor.scores(rRUM)
write.csv(target, 'G4_lee/698/label.csv')

q = read.csv("SFS/536/q.csv")[-1]
resp = read.csv("SFS/536/resp.csv")[-1]
LCDM = gdina(resp, q.matrix=q, link="logit", maxit=maxit, progress=progress)
target = IRT.factor.scores(LCDM)
write.csv(target, 'FSF/536/label.csv')