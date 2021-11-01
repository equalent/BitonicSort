#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <vector>

#include <amp.h>

namespace amp = concurrency;
using ViewType = amp::array_view<int, 1>;

#define SEQ_SIZE 4096

void bitonicAmp(amp::accelerator_view acc_v, std::vector<int>& data)
{
    amp::extent<1> ex(data.size());

    amp::array<int, 1> gpuData(ex, acc_v, amp::access_type_read_write);
    memcpy_s(gpuData.data(), data.size() * sizeof(int), data.data(), data.size() * sizeof(int));

    amp::array_view<int, 1> view(gpuData);

    for (int k = 2; k <= (int)data.size(); k *= 2)
    {
        for (int j = k / 2; j > 0; j /= 2)
        {
            amp::parallel_for_each(amp::extent<1>(data.size()), [=](amp::index<1> idx) restrict(amp) {
                int i = idx[0];

                int l = i ^ j;
                amp::index<1> lIdx(l);
                if (l > i)
                {
                    if ((((i & k) == 0) && (view[idx] > view[lIdx])) || (((i & k) != 0) && (view[idx] < view[lIdx])))
                    {
                        int tmp = view[idx];
                        view[idx] = view[lIdx];
                        view[lIdx] = tmp;
                    }
                }
            });
        }
    }

    amp::array_view<int, 1> cpuView(data);
    gpuData.copy_to(cpuView);
}

void bitonicCpu(std::vector<int>& data)
{
    for (int k = 2; k <= (int)data.size(); k *= 2)
    {
        for (int j = k / 2; j > 0; j /= 2)
        {
            for (int i = 0; i < (int)data.size(); i++)
            {
                int l = i ^ j;
                if (l > i)
                {
                    if ((((i & k) == 0) && (data[i] > data[l])) || (((i & k) != 0) && (data[i] < data[l])))
                    {
                        int tmp = data[i];
                        data[i] = data[l];
                        data[l] = tmp;
                    }
                }
            }
        }
    }
}

bool checkSorted(const std::vector<int>& data, const char* sortType)
{
    for (int i = 0; i < (int)data.size(); i++)
    {
        if (i > 0)
        {
            if (data[i] < data[i - 1])
            {
                printf("[ERROR] %s result not sorted!\n", sortType);
                return false;
            }
        }
    }
    return true;
}

int main()
{
    amp::accelerator acc = amp::accelerator(amp::accelerator::default_accelerator);
    std::wstring accDesc = acc.get_description();

    wprintf_s(L"[Bitonic Sort AMP Implementation]\nUsing accelerator: %s\n", accDesc.c_str());

    acc.set_default_cpu_access_type(amp::access_type_read_write);
    amp::accelerator_view acc_v = acc.get_default_view();

    printf("Sorting an array of %d elements...\n", (int)SEQ_SIZE);

    std::vector<int> dataAmp(SEQ_SIZE);
    time_t t;

    printf("Generating random data...\n\n");
    srand((unsigned)time(&t));
    for (int& v : dataAmp)
    {
        v = rand() % 4096;
    }

    std::vector<int> dataCpu(dataAmp);

    printf("Sorting with AMP...\n");
    clock_t tAmpBegin = clock();
    bitonicAmp(acc_v, dataAmp);
    clock_t tAmpEnd = clock();
    clock_t tAmp = tAmpEnd - tAmpBegin;

    printf("AMP took: %lf\n\n", double(tAmp) / double(CLOCKS_PER_SEC));
    if (!checkSorted(dataAmp, "AMP"))return -1;

    printf("Sorting with CPU...\n");
    clock_t tCpuBegin = clock();
    bitonicCpu(dataCpu);
    clock_t tCpuEnd = clock();
    clock_t tCpu = tCpuEnd - tCpuBegin;

    printf("CPU took: %lf\n\n", double(tCpu) / double(CLOCKS_PER_SEC));
    if (!checkSorted(dataCpu, "CPU"))return -1;

    return 0;
}