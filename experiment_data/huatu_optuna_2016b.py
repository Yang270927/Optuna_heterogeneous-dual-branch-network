import matplotlib.pyplot as plt

# optuna 0.6628
y1 = [0.1142,0.1303,0.1515,0.2059,0.3087,0.3840,0.4861,0.6578,0.7527,0.8520,0.8974,
      0.9142,0.9188,0.9276,0.9228,0.9319,0.9276,0.9302,0.9271,0.9301]
average1 = sum(y1) / len(y1)
print(f"均值为: {average1:.4f}")
#DAE 0.6411
y2 = [0.1090,0.1085,0.1259,0.1619,0.2276,0.3059,0.4145,0.5465,0.7033,0.8343,0.8990,
      0.9210,0.9292,0.9324,0.9333,0.9331,0.9350,0.9350,0.9344,0.9318]
average2 = sum(y2) / len(y2)
print(f"均值为: {average2:.4f}")
#MCLDNN 0.6444
y3 = [0.1138,0.1099,0.1199,0.1451,0.2129,0.3021,0.4183,0.5673,0.7287,0.8559,0.9122,
      0.9260,0.9321,0.9342,0.9340,0.9333,0.9361,0.9345,0.9354,0.9365]
average3 = sum(y3) / len(y3)
print(f"均值为: {average3:.4f}")
#PET_CGDNN 0.6360
y4 = [0.1025,0.1088,0.1234,0.1510,0.2231,0.3046,0.3940,0.5403,0.7007,0.8313,0.8960,
      0.9172,0.9248,0.9254,0.9277,0.9324,0.9300,0.9288,0.9283,0.9295]
average4 = sum(y4) / len(y4)
print(f"均值为: {average4:.4f}")
#MHNN 0.6487
y5 = [0.1082,0.1160,0.1310,0.1778,0.2426,0.3495,0.4433,0.6215,0.7497,0.8230,0.8844,
      0.9093,0.9160,0.9259,0.9262,0.9219,0.9258,0.9188,0.9247,0.9222]
average5 = sum(y5) / len(y5)
print(f"均值为: {average5:.4f}")
#WACN 0.6443
y6 = [0.1080,0.1066,0.1203,0.1547,0.2008,0.2876,0.4177,0.5912,0.7480,0.8449,0.9138,
      0.9257,0.9290,0.9331,0.9345,0.9339,0.9371,0.9368,0.9371,0.9392]
average6 = sum(y6) / len(y6)
print(f"均值为: {average6:.4f}")
x = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18]
plt.figure(figsize=(6, 6))

plt.plot(x, y1, linewidth=1, color="red", marker="o")
plt.plot(x, y2, linewidth=1, color="orange", marker="o")
plt.plot(x, y3, linewidth=1, color="blue", marker="o")
plt.plot(x, y4, linewidth=1, color="green", marker="o")
plt.plot(x, y5, linewidth=1, color="purple", marker="o")
plt.plot(x, y6, linewidth=1, color="cyan", marker="o")



plt.xlabel("SNR(dB)")
plt.ylabel("accuracy")
# plt.title(f'recognition accuracy of RML2016.10b')

plt.legend(["optuna","DAE","MCLDNN","PET_CGDNN","MHNN","WACN"], loc="lower right")

plt.xlim([-20, 20])
plt.ylim([0, 1])

plt.grid()
acc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
plt.yticks(acc)
plt.xticks(x)

plt.show()

