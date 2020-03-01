import string
import random
import time
import numpy as np
import pyopencl as cl  
import math
# You can input here
length = 16
num = 50000
randStr = "abcd34gawegdf123443546t34ergegf25gfbdsvssfv"
# randStr = "abcdefghijklmnopqrstuvwxyz0123456789"

# end of Input


def randomwords(size, chars):
	return ''.join(random.choice(chars) for _ in range(size))

def cpu_gen(length, num, randStr):
	str_list=[]
	cpu_start_time = time.time()  # Get the CPU start time
	for i in range(num):
		str_list.append(randomwords(length, randStr))
	cpu_end_time = time.time()  # Get the CPU end time
	timeGap = cpu_end_time - cpu_start_time
	print("CPU Time: {0} s".format(timeGap))
	print("****************************************************")
	print("Random Strings gnerated by CPU")
	print(str_list)
	return timeGap

def gpu_gen(length, num, randStr):
	#define random basic letters
	# radom base length
	totChar = length * num
	randomIndex = len(randStr)
	# parameters of random generation function
	# right_index = np.array.randint(randomIndex, size= totChar)
	right_index = np.empty(totChar, dtype=str)
	ranss = np.array((randStr,))

	rand_seed = math.floor(time.time()) % 20
	# define the contect to use gpu
	ctx = cl.create_some_context()
	queue = cl.CommandQueue(ctx)
	# define flag of the opencl
	mf = cl.mem_flags

	rand_buff_1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=right_index)
	rand_str = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ranss)
	program = cl.Program(ctx, """

	int mpow(int n, int p){
		int s =1;
		for(int i=1; i<=p; ++i){
			s = s * n;
		}
		return s;
	}

	int myRand(int next){
		next = ((next * (next % 3) )/50)% 100;
		return next;
	}

	int mabs(int a){
		if (a<0){
			a = 0-a;
		}
		return a;
	}

	int myRandInRange(int next, int min, int max){
		int temp;
		if (max < min){
			temp = max;
			max = min;
			min = temp;
		}
		return myRand(next) % (max + 1 - min) + min;
	}


	__kernel void RandGen(
	    __global int *rand_out_buff, __global char *ss, int rand_buff, int seed_buff)
	{
	  
	  int gid = get_global_id(0);
	 
	  int temp = mpow(gid, seed_buff) + seed_buff % (gid+1);
	  
	  int input = myRandInRange(temp, gid * seed_buff, gid * (gid % seed_buff)) + seed_buff;
	  input  = mabs(input);
	  rand_out_buff[gid] =ss[(input % rand_buff)*4];
	}
	""").build()

	rand_out_buff = cl.Buffer(ctx, mf.WRITE_ONLY, right_index.nbytes)
	gpu_start_time = time.time()
	event = program.RandGen(queue, right_index.shape, None, rand_out_buff, rand_str, np.int32(randomIndex), np.int32(rand_seed))
	event.wait()
	# elapsed = 1e-9*(event.profile.end - event.profile.start)
	rand_list = np.empty_like(right_index)
	cl.enqueue_copy(queue, rand_list, rand_out_buff)
	gpu_end_time = time.time()  # Get the GPU end time
	timeGap = gpu_end_time - gpu_start_time
	print("GPU Time: {0} s".format(timeGap))
	for i in range(totChar):
		if i>1 and i % length==0:print(', ', end='')
		print(rand_list[i], end='')
	print('')
	print("****************************************************")
	print("Random Strings gnerated by GPU")
	return timeGap

	cpu_time = cpu_gen(length, num, randStr)
gpu_time = gpu_gen(length, num, randStr)
print("****************************************************")
print("CPU and GPU execution time comparison")
print("CPU Time: {0} s".format(cpu_time))
print("GPU Time: {0} s".format(gpu_time))