from utils import *
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from config import *
import math

source = f'data/{language}/train/code.original_subtoken'
target = f'data/{language}/train/javadoc.original'
train_data = read_parallel((source, target), 0, 0, math.inf, math.inf)

fun_len = np.zeros(len(train_data))
com_len = np.zeros(len(train_data))

for idx, (fun, com) in enumerate(train_data):
    fun_len[idx] = len(fun)
    com_len[idx] = len(com)

print("Avg. Function Length: {:.2f} ({:.2f}-{:.2f})".format(np.mean(fun_len), np.min(fun_len), np.max(fun_len)))
print("Avg. Javadoc Length: {:.2f} ({:.2f}-{:.2f})".format(np.mean(com_len), np.min(com_len), np.max(com_len)))
print("Pearson Correlation: {:.4f}, p-value: {:.4f}".format(*pearsonr(fun_len, com_len)))

model = LinearRegression()
model.fit(fun_len.reshape(-1, 1), com_len)
print(f"The best fit line is com_len = {model.coef_[0]:.4f} * fun_len + {model.intercept_:.4f}")

print("Max. Function Length: {:.2f}".format(np.max(fun_len)))
print("Max. Javadoc Length: {:.2f}".format(np.max(com_len)))

print("0.999 quantile Function Length: {:.2f}".format(np.quantile(fun_len, 0.999)))
print("0.999 quantile Javadoc Length: {:.2f}".format(np.quantile(com_len, 0.999)))
