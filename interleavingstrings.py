# x = "chocolate"
# y = "chips"
# z = "cchocohilaptes"

x = "1111"
y = "11"
z = "111111"

result = {}

result[(0, 0, 0)] = True

for j in range(1, len(y) + 1):
    result[(0, j, j)] = (y[j-1] == z[j-1]) and result[(0, j-1, j-1)]
for i in range(1, len(x) + 1):
    result[(i, 0, i)] = x[i-1] == z[i-1] and result[(i-1, 0, i-1)]


for i in range(1, len(x) + 1):
    for j in range(1, len(y) + 1):
        result[(i, j, i + j)] = ((x[i-1] == z[i + j -1]) and result[(i - 1, j, i + j - 1)]) or ((y[j-1] == z[i + j-1]) and result[(i, j-1, i + j - 1)])

print(result[(len(x), len(y), len(z))])



