"""
Módulo para cálculo de métricas de eficiência computacional.
"""

import torch
import time
import numpy as np
from typing import Dict


def measure_flops(model, device, input_size=(1, 3, 224, 224)) -> Dict[str, float]:
    """
    Calcula FLOPs (Floating Point Operations) do modelo.
    
    Args:
        model: Modelo PyTorch
        device: Dispositivo (cuda/cpu)
        input_size: Tamanho do tensor de entrada
        
    Returns:
        Dicionário com FLOPs e parâmetros
    """
    try:
        from thop import profile
        
        input_tensor = torch.randn(input_size).to(device)
        model.eval()
        
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        
        return {
            'flops': flops,
            'gflops': flops / 1e9,
            'mflops': flops / 1e6,
            'params': params,
            'params_millions': params / 1e6
        }
    except ImportError:
        print("⚠ Biblioteca 'thop' não instalada. Instale com: pip install thop")
        return {}


def measure_latency(model, device, input_size=(1, 3, 224, 224), 
                   num_warmup=100, num_iterations=1000) -> Dict[str, float]:
    """
    Mede latência (tempo de inferência) do modelo.
    
    Args:
        model: Modelo PyTorch
        device: Dispositivo
        input_size: Tamanho da entrada
        num_warmup: Número de iterações de warmup
        num_iterations: Número de iterações para medição
        
    Returns:
        Dicionário com estatísticas de latência (ms)
    """
    model.eval()
    input_tensor = torch.randn(input_size).to(device)
    
    # Warmup
    print(f"  Executando {num_warmup} iterações de warmup...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    
    # Sincronizar GPU se disponível
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Medição
    print(f"  Medindo latência em {num_iterations} iterações...")
    times = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.time()
            _ = model(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.time()
            times.append((end - start) * 1000)  # Converter para ms
    
    times = np.array(times)
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99)
    }


def measure_memory(model, device, input_size=(1, 3, 224, 224)) -> Dict[str, float]:
    """
    Mede consumo de memória GPU durante inferência.
    
    Args:
        model: Modelo PyTorch
        device: Dispositivo
        input_size: Tamanho da entrada
        
    Returns:
        Dicionário com uso de memória (MB)
    """
    if device.type != 'cuda':
        print("⚠ Medição de memória disponível apenas para CUDA")
        return {}
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model.eval()
    input_tensor = torch.randn(input_size).to(device)
    
    with torch.no_grad():
        _ = model(input_tensor)
    
    memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)  # MB
    
    return {
        'memory_allocated_mb': memory_allocated,
        'memory_reserved_mb': memory_reserved
    }


def measure_model_size(model) -> Dict[str, float]:
    """
    Calcula tamanho do modelo serializado.
    
    Args:
        model: Modelo PyTorch
        
    Returns:
        Dicionário com tamanho em diferentes unidades
    """
    import io
    
    # Salvar modelo em buffer
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.tell()
    
    return {
        'size_bytes': size_bytes,
        'size_kb': size_bytes / 1024,
        'size_mb': size_bytes / (1024 ** 2),
        'size_gb': size_bytes / (1024 ** 3)
    }


def benchmark_model(model, device, verbose=True) -> Dict[str, any]:
    """
    Executa benchmark completo do modelo.
    
    Args:
        model: Modelo PyTorch
        device: Dispositivo
        verbose: Se deve imprimir resultados
        
    Returns:
        Dicionário com todas as métricas de eficiência
    """
    print("\n" + "="*60)
    print("BENCHMARK DE EFICIÊNCIA COMPUTACIONAL")
    print("="*60)
    
    results = {}
    
    # FLOPs
    print("\n[1/4] Calculando FLOPs...")
    flops_info = measure_flops(model, device)
    results['flops'] = flops_info
    if verbose and flops_info:
        print(f"  FLOPs: {flops_info['gflops']:.2f} G")
        print(f"  Parâmetros: {flops_info['params_millions']:.2f} M")
    
    # Latência
    print("\n[2/4] Medindo latência...")
    latency_info = measure_latency(model, device)
    results['latency'] = latency_info
    if verbose:
        print(f"  Latência média: {latency_info['mean_ms']:.2f} ± {latency_info['std_ms']:.2f} ms")
        print(f"  P95: {latency_info['p95_ms']:.2f} ms | P99: {latency_info['p99_ms']:.2f} ms")
    
    # Memória
    print("\n[3/4] Medindo consumo de memória...")
    memory_info = measure_memory(model, device)
    results['memory'] = memory_info
    if verbose and memory_info:
        print(f"  Memória alocada: {memory_info['memory_allocated_mb']:.2f} MB")
    
    # Tamanho do modelo
    print("\n[4/4] Calculando tamanho do modelo...")
    size_info = measure_model_size(model)
    results['size'] = size_info
    if verbose:
        print(f"  Tamanho do modelo: {size_info['size_mb']:.2f} MB")
    
    print("\n" + "="*60 + "\n")
    
    return results


def print_efficiency_comparison(results_model1: Dict, results_model2: Dict,
                                model1_name: str = "Model 1", 
                                model2_name: str = "Model 2") -> None:
    """
    Imprime comparação de eficiência entre dois modelos.
    
    Args:
        results_model1: Resultados do benchmark do modelo 1
        results_model2: Resultados do benchmark do modelo 2
        model1_name: Nome do modelo 1
        model2_name: Nome do modelo 2
    """
    print("\n" + "="*70)
    print("COMPARAÇÃO DE EFICIÊNCIA")
    print("="*70)
    print(f"{'Métrica':<30} {model1_name:>15} {model2_name:>15} {'Diferença':>10}")
    print("-"*70)
    
    # FLOPs
    if 'flops' in results_model1 and 'flops' in results_model2:
        flops1 = results_model1['flops']['gflops']
        flops2 = results_model2['flops']['gflops']
        diff = ((flops2 - flops1) / flops1) * 100
        print(f"{'FLOPs (G)':<30} {flops1:>15.2f} {flops2:>15.2f} {diff:>9.1f}%")
    
    # Latência
    if 'latency' in results_model1 and 'latency' in results_model2:
        lat1 = results_model1['latency']['mean_ms']
        lat2 = results_model2['latency']['mean_ms']
        diff = ((lat2 - lat1) / lat1) * 100
        print(f"{'Latência (ms)':<30} {lat1:>15.2f} {lat2:>15.2f} {diff:>9.1f}%")
    
    # Memória
    if 'memory' in results_model1 and 'memory' in results_model2:
        if results_model1['memory'] and results_model2['memory']:
            mem1 = results_model1['memory']['memory_allocated_mb']
            mem2 = results_model2['memory']['memory_allocated_mb']
            diff = ((mem2 - mem1) / mem1) * 100
            print(f"{'Memória (MB)':<30} {mem1:>15.2f} {mem2:>15.2f} {diff:>9.1f}%")
    
    # Tamanho
    if 'size' in results_model1 and 'size' in results_model2:
        size1 = results_model1['size']['size_mb']
        size2 = results_model2['size']['size_mb']
        diff = ((size2 - size1) / size1) * 100
        print(f"{'Tamanho (MB)':<30} {size1:>15.2f} {size2:>15.2f} {diff:>9.1f}%")
    
    print("="*70 + "\n")