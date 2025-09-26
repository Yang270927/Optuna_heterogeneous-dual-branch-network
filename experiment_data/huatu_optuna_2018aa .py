import matplotlib.pyplot as plt

# optuna  0.5930  366281
y1 = [0.1548,0.1536,0.1522,0.1731,0.1977,0.2590,0.3276,0.4414,0.5621,0.6787,0.7804,
      0.8298,0.8547,0.8576,0.8580,0.8713,0.8701,0.8676,0.8555,0.8599,0.8572]
average1 = sum(y1) / len(y1)
print(f"均值为: {average1:.4f}")
#DAE  0.5526   14778
y2 = [0.1412,0.1437,0.1425,0.1637,0.2200,0.2918,0.3631,0.4106,0.4731,0.5687,0.6793,
      0.7662,0.7818,0.8000,0.7981,0.7968,0.8056,0.7993,0.8100,0.8025,0.8168]
average2 = sum(y2) / len(y2)
print(f"均值为: {average2:.4f}")
#MCLDNN 0.6003  404788  39693.33 秒
y3 = [0.1112,0.1387,0.1312,0.1718,0.2256,0.2900,0.3856,0.4650,0.5400,0.6618,0.7481,
      0.8212,0.8550,0.8718,0.8793,0.8812,0.8831,0.8837,0.8900,0.8843,0.8893]
average3 = sum(y3) / len(y3)
print(f"均值为: {average3:.4f}")
#PET_CGDNN 0.5887
y4 = [0.1237,0.1318,0.1293,0.1375,0.1768,0.2225,0.3306,0.4400,0.5262,0.6468,0.7812,
      0.8325,0.8593,0.8687,0.8675,0.8843,0.8768,0.8868,0.8743,0.8843,0.8837]
average4 = sum(y4) / len(y4)
print(f"均值为: {average4:.4f}")
#MHNN  0.5134 366281
y5 = [0.1273,0.1386,0.1365,0.1467,0.1777,0.2140,0.2754,0.3553,0.4816,0.5262,0.5897,
      0.6539,0.7304,0.7579,0.7753,0.7966,0.7944,0.7788,0.7870,0.7859,0.7797]
average5 = sum(y5) / len(y5)
print(f"均值为: {average5:.4f}")
#WACN 0.6027 
y6 = [0.1422,0.1308,0.1468,0.1600,0.2026,0.2941,0.3657,0.4194,0.5398,0.6278,0.7690,
      0.8507,0.8766,0.8850,0.8792,0.8680,0.8832,0.8790,0.8712,0.8876,0.8883]
average6 = sum(y6) / len(y6)
print(f"均值为: {average6:.4f}")
# average = sum(y6) / len(y6)
# print(f"均值为: {average:.4f}")
x = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]
plt.figure(figsize=(6, 6))

plt.plot(x, y1, linewidth=1, color="red", marker="o")
plt.plot(x, y2, linewidth=1, color="pink", marker="o")
plt.plot(x, y3, linewidth=1, color="blue", marker="o")
plt.plot(x, y4, linewidth=1, color="green", marker="o")
plt.plot(x, y5, linewidth=1, color="purple", marker="o")
plt.plot(x, y6, linewidth=1, color="orange", marker="o")



plt.xlabel("SNR(dB)")
plt.ylabel("accuracy")
# plt.title(f'recognition accuracy of RML2018.10aa')

plt.legend(["optuna","DAE","MCLDNN","PET_CGDNN","MHNN","WACN"], loc="lower right")

plt.xlim([-20, 20])
plt.ylim([0, 1])

plt.grid()
acc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
plt.yticks(acc)
plt.xticks(x)

plt.show()

