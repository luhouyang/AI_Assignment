cxstr = ""
CXPB = 1.0
for i in range(19):
    cxstr += str(round(CXPB, 3)) + ','

    mustr = ""
    MU_INDPB = 0.01
    for i in range(13):
        mustr += str(round(MU_INDPB, 3)) + ','
        MU_INDPB += 0.02

    CXPB -= 0.05

print(mustr)
print(cxstr)