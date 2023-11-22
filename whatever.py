

input = [0, -1, 1, 1, -1]

input_len = 5

totalover0 = 0
totallow0 = 0

for i in range(input_len):
    if input[i] > 0:
        totalover0 += 1

    elif input[i] < 0:
        totallow0 += 1

total0 = input_len-totalover0-totalover0

EL = 0
if input[0] <= 0:
    ER = input_len-totalover0
else:
    ER = input_len-totalover0+1

lastN = input[0]

minE = input_len
for i in range(1, input_len):
    current = input[i]
    if lastN >= 0:
        EL += 1
    if current <= 0:
        ER -= 1
    lastN = current
    E = EL+ER
    if E < minE:
        minE = E
print(minE)
