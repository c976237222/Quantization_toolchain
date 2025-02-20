print('正计算网络量化误差(SNR)，最后一层的误差应小于 0.1 以保证量化精度:')
reports = graphwise_error_analyse(
    graph=quantized, running_device="cuda", steps=32,
    dataloader=dataloader, collate_fn=lambda x: x.to("cuda"))
for op, snr in reports.items():
    if snr > 0.1: ppq_warning(f'层 {op} 的累计量化误差显著，请考虑进行优化')

if REQUIRE_ANALYSE:
    print('正计算逐层量化误差(SNR)，每一层的独立量化误差应小于 0.1 以保证量化精度:')
    layerwise_error_analyse(graph=quantized, running_device="cuda",
                            interested_outputs=None,
                            dataloader=dataloader, collate_fn=lambda x: x.to("cuda"))