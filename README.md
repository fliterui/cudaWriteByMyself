不会

```c++
__global__ void shit(int *a)
{
    printf("%d",*a);
}

void useShit(int* a)
{
    shit<<<1,1>>>(a);
}

int main()
{
    int a, *d_a;
    cudaMalloc((void**)&d_a, sizeof(int));
    a = 12345;
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    useShit(d_a);
    return 0;
}
说明可以直接传cudaMalloc的值
