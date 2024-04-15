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

/*那个cfar的函数感觉可以抢救一下, 再改改啥的
* warmup
* 你别说, 传数据也可以异步, 把signal分成几块, 弄多个流, 同时传是不是能快点
* 记得提一下那个结果的纵坐标是干啥的, 咋给他修一下, 整的都好几十万的值, 就离谱
* 好像cudaMalloc ori和cudaMemcpy ori可以和pc的d_signal的fft并行
* 块和网格结构 
* 瘦块
* 看看那些规约, 循环展开
* 读数据(特别是转置), 还有延迟隐藏
* 还有指令的延迟隐藏
* 师兄的库
* 流
* nsight
* 共享内存有用吗?
* 师兄发的那个fpga的论文
* 那个原子级啥啥啥的有用吗		->这一章涉及了精确度
* 内存类型
*/
