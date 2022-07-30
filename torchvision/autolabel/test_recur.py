def recur(arr, outp, index):
	if not isinstance(arr, list):
		outp[index] = arr
	else:
		if not isinstance(arr[0], list):
			last = True
		else:
			last = False
		for i in range(len(arr)):
			outp.append([])
			deeper = outp[i]
			if last:
				deeper = outp
			recur(arr[i], deeper, i)


arr = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]]
outp = []
new_arr = recur(arr, outp, 0)
print(arr)
print(outp)
print(f'test status: {arr == outp}')


