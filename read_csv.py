import matplotlib.pyplot as plt
import csv
import numpy as np
import config
def PlotCsv(f):
	reader = csv.DictReader(f)
	xaxis = []
	exposed = []
	infected = []
	recovered = []
	susceptible = []
	quarantined = []

	for i,row in enumerate(reader):
		xaxis.append(i)
		exposed.append(eval(row['exposed']))
		infected.append(eval(row['infected']))
		recovered.append(eval(row['recovered']))
		susceptible.append(eval(row['susceptible']))
		quarantined.append(eval(row['quarantined']))


	susceptible,infected,recovered,quarantined,exposed = np.array(susceptible),np.array(infected),np.array(recovered),np.array(quarantined),np.array(exposed)
	plt.plot(xaxis,susceptible/10000)
	plt.plot(xaxis,infected/10000)
	plt.plot(xaxis,recovered/10000)
	plt.plot(xaxis,quarantined/10000)
	plt.plot(xaxis,exposed/10000)
	plt.legend(['susceptible','infected','recovered','quarantined','exposed'])
	plt.xlabel('Days')
	plt.ylabel('fraction of Population')
	plt.show()

def PlotCsvvsGamma(f_arr):
	xaxis = []
	# gama_1 = []
	# gama_2 = []
	# gama_3 = []
	# gama_4 = []
	gama = []

	for f in f_arr:
		reader = csv.DictReader(f)
		arr = []
		for i,row in enumerate(reader):
			arr.append((eval(row['exposed'])+eval(row['infected']))/config.N)
		print(len(arr))
		gama.append(arr)

	gama_0,gama_1,gama_2,gama_3,gama_4 = gama
	min_ = min(len(gama_0),len(gama_1),len(gama_2),len(gama_3),len(gama_4))
	for i in range(min_):
		xaxis.append(i+1)


	plt.plot(xaxis,gama_0[:min_])
	plt.plot(xaxis,gama_1[:min_])
	plt.plot(xaxis,gama_2[:min_])
	plt.plot(xaxis,gama_3[:min_])
	plt.plot(xaxis,gama_4[:min_])
	plt.legend(['0.0','0.1','0.2','0.3','0.4'])
	plt.xlabel('Days')
	plt.ylabel('Exposed + Infected')
	plt.show()


if __name__=="__main__":
	f = [open('out_gamma_0.csv',newline=''),open('out_gamma_1.csv',newline=''),open('out_gamma_2.csv',newline=''),open('out_gamma_3.csv',newline=''),open('out_gamma_4.csv',newline='')]
	PlotCsvvsGamma(f)
	# PlotCsovLog(f)
