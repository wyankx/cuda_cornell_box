# Рендер CornellBox с помощью CUDA

Репозиторий создан из [bhs-quake-rtx-metal-raytracing-shaders](https://github.com/smorf1koo/bhs-quake-rtx-metal-raytracing-shaders) для выполнения задания

На данный момент реализован в формате compute, без ray-tracing (может потом что нибудь добавлю, а может и нет)

# Компиляция и запуск
Компиляция:
```
nvcc starter/cuda/main.cu
```

Запуск:
```
./a.out
```

# Скорость работы
На GeForce RTX 2060 Super рендерит 512x512 с 256 фреймами за 0.15-0.18 с.
