from irgan.inference.test import Tester
from irgan.evaluation.evaluate_metrics import report_inception_objects_score

if __name__ == '__main__':
    cfg = parse_config()
    tester = Tester(cfg, test_eval=True)
    tester.test()
    del tester
    torch.cuda.empty_cache()
    metrics_report = dict()
    if cfg.metric_inception_objects:
        io_jss, io_ap, io_ar, io_af1, io_cs, io_gs = report_inception_objects_score(
            None, None, None, cfg.results_path,
            keys[cfg.dataset + '_inception_objects'], keys[cfg.test_dataset],
            cfg.dataset)

        metrics_report['jaccard'] = io_jss
        metrics_report['precision'] = io_ap
        metrics_report['recall'] = io_ar
        metrics_report['f1'] = io_af1
        metrics_report['cossim'] = io_cs
        metrics_report['relsim'] = io_gs
    print(metrics_report)
