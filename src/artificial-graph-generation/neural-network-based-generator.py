from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from math import inf
import os
import time
from typing import Dict
import torch
from pathlib import Path
from generators.generation_gan_autoregressive import GraphGenGanDescriminator, GraphGenGanGenerator
from generators.generation_lstm_autoregressive import GraphGenLSTMGenerator, GraphGenLSTMDescriminator
from generators.generation_lstm_autoregressive_softmax import GraphGenLSTMGeneratorNodeSoftmax, GraphGenLSTMDescriminatorSoftmax
from generators.generation_one_shot_linear import GraphGenLinearDescriminator, GraphGenLinearGenerator
from generators.utils import *
from tqdm import tqdm
import teneto
import torch.nn.functional as F
import pandas
import overtime
from overtime.algorithms import temporal_betweenness, temporal_degree, temporal_closeness, temporal_pagerank

num_nodes = 17
nodes = ['carts', 'carts-db', 'catalogue', 'catalogue-db', 'front-end', 'orders', 'orders-db', 'payment', 'queue-master', 'rabbitmq', 'session-db', 'shipping', 'user', 'user-db', 'worker1', 'worker2', 'master']

def trainGraphGenTemporalSequence(
    cfg: Dict,
    dataset="../generated/100k_frontend_graphs-temporal_unique.npy",
    outdir=Path("./output") / "LSTMAutoregressiveSequence",
    generator = GraphGenLSTMGeneratorNodeSoftmax(num_nodes, torch.nn.Tanh, 18)
):
    outdir = Path(outdir) / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(outdir, exist_ok=False)
    num_nodes = cfg["num_nodes"]
    time_steps = cfg["time_steps"]
    evaluate_size = cfg["evaluate_size"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    step = cfg["step"]
    clip=cfg["clip"]
    clip_value=cfg["clip_value"]

    dataset_tb = np.mean(np.load("generated/temporal_betweennesses_unique.npy"),axis=0)
    dataset_tc = np.mean(np.load("generated/temporal_closenesses_unique.npy"),axis=0)
    dataset_td = np.mean(np.load("generated/temporal_degrees_unique.npy"),axis=0)
    dataset_tbv = np.var(np.load("generated/temporal_betweennesses_unique.npy"),axis=0)
    dataset_tcv = np.var(np.load("generated/temporal_closenesses_unique.npy"),axis=0)
    dataset_tdv = np.var(np.load("generated/temporal_degrees_unique.npy"),axis=0)

    best_loss = inf
    t1 = time.perf_counter()
    loader = get_sequence_dataloader(dataset, batch_size)
    print(f"loaded data in {time.perf_counter() - t1:.2f}s")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    lr = cfg["lr"]
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    with open(outdir/"setup.txt", "w") as write_handle:
        write_handle.write(repr(generator))
        write_handle.write(f"\nDataset: {dataset}\n")
        write_handle.write("\n".join([f"{key}: {item}" for key, item in cfg.items()]))
    log = {
        "loss": [],
    }
    mean_differences = []
    tbs = []
    tds = []
    tcs = []
    tbvs = []
    tdvs = []
    tcvs = []
    tbsdiff = []
    tdsdiff = []
    tcsdiff = []
    tbvsdiff = []
    tdvsdiff = []
    tcvsdiff = []
    while step <= epochs:
        bar = tqdm(loader, desc=f"[{step}/{epochs}] train")
        for graph_batch in bar:
            generator.train()
            generator_optimizer.zero_grad()
            random_data = torch.zeros((graph_batch.shape[0], (2*(num_nodes+1)))).float()
            generated: torch.Tensor = generator.sequence_forward(random_data, graph_batch, graph_batch.shape[1])
            loss: torch.Tensor = F.cross_entropy(generated.reshape(-1, graph_batch.shape[-1]), graph_batch.reshape(-1, graph_batch.shape[-1]))
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_value)
            generator_optimizer.step()
            log["loss"].append(loss.detach().cpu().item())
            bar.set_postfix({
                "loss": loss
            })
            if loss.detach().cpu().item() < best_loss:
                best_loss = loss.detach().cpu().item()
                torch.save(generator.state_dict(), outdir/"best-loss-generator.ph")
        # Evaluate
        if step % 1 == 0: 
            with torch.no_grad():
                random_data = torch.zeros((graph_batch.shape[0], (2*(num_nodes+1)))).float()
                generated = generator(random_data)
                generated = sequence_to_adj(generated[:,:-1,:])
                tb, tc, td, tbv, tcv, tdv = get_temporal_metrics(generated, all=(step%20 == 0))
                tbs.append({nodes[i]: tb[i] for i in range(len(nodes))})
                tcs.append({nodes[i]: tc[i] for i in range(len(nodes))})
                tds.append({nodes[i]: td[i] for i in range(len(nodes))})
                tbvs.append({nodes[i]: tbv[i] for i in range(len(nodes))})
                tcvs.append({nodes[i]: tcv[i] for i in range(len(nodes))})
                tdvs.append({nodes[i]: tdv[i] for i in range(len(nodes))})
                tbdiff, tcdiff, tddiff = dataset_tb - tb, dataset_tc - tc, dataset_td - td
                tbvdiff, tcvdiff, tdvdiff = dataset_tbv - tbv, dataset_tcv - tcv, dataset_tdv - tdv
                tbsdiff.append({nodes[i]: tbdiff[i] for i in range(len(nodes))})
                tcsdiff.append({nodes[i]: tcdiff[i] for i in range(len(nodes))})
                tdsdiff.append({nodes[i]: tddiff[i] for i in range(len(nodes))})
                tbvsdiff.append({nodes[i]: tbvdiff[i] for i in range(len(nodes))})
                tcvsdiff.append({nodes[i]: tcvdiff[i] for i in range(len(nodes))})
                tdvsdiff.append({nodes[i]: tdvdiff[i] for i in range(len(nodes))})
                print(tbdiff, tcdiff, tddiff)
                print(tbvdiff, tcvdiff, tdvdiff)
                diffsum = abs(tbdiff) + abs(tcdiff) + abs(tddiff) + abs(tbvdiff) + abs(tcvdiff) + abs(tdvdiff)
                mean_differences.append(diffsum.sum())
            if diffsum.sum() == min(mean_differences):
                torch.save(generator.state_dict(), outdir/"best-generator.ph")
            tbdf = pandas.DataFrame.from_records(tbs, columns=nodes)
            tcdf = pandas.DataFrame.from_records(tcs, columns=nodes)
            tddf = pandas.DataFrame.from_records(tds, columns=nodes)

            tbdf.to_csv(outdir/"temporal-betweenesses-mean.csv")
            tcdf.to_csv(outdir/"temporal-clonsesses-mean.csv")
            tddf.to_csv(outdir/"temporal-degrees-mean.csv")

            tbdiffdf = pandas.DataFrame.from_records(tbsdiff, columns=nodes)
            tcdiffdf = pandas.DataFrame.from_records(tcsdiff, columns=nodes)
            tddiffdf = pandas.DataFrame.from_records(tdsdiff, columns=nodes)
            

            tbdiffdf.to_csv(outdir/"temporal-betweenesses-mean-differences.csv")
            tcdiffdf.to_csv(outdir/"temporal-clonsesses-mean-differences.csv")
            tddiffdf.to_csv(outdir/"temporal-degrees-mean-differences.csv")

            tbvdf = pandas.DataFrame.from_records(tbvs, columns=nodes)
            tcvdf = pandas.DataFrame.from_records(tcvs, columns=nodes)
            tdvdf = pandas.DataFrame.from_records(tdvs, columns=nodes)

            tbvdf.to_csv(outdir/"temporal-betweenesses-var.csv")
            tcvdf.to_csv(outdir/"temporal-clonsesses-var.csv")
            tdvdf.to_csv(outdir/"temporal-degrees-var.csv")

            tbvdiffdf = pandas.DataFrame.from_records(tbvsdiff, columns=nodes)
            tcvdiffdf = pandas.DataFrame.from_records(tcvsdiff, columns=nodes)
            tdvddifff = pandas.DataFrame.from_records(tdvsdiff, columns=nodes)
            

            tbvdiffdf.to_csv(outdir/"temporal-betweenesses-var-differences.csv")
            tcvdiffdf.to_csv(outdir/"temporal-clonsesses-var-differences.csv")
            tdvddifff.to_csv(outdir/"temporal-degrees-var-differences.csv")

            
            plt.figure(figsize=(10,5))
            tbdiffdf.plot(ax=plt.gca(), title="Temporal Betweeness over Time",xlabel="step", ylabel="mean difference")
            plt.savefig(outdir/"temporal-betweeness-mean-differences.svg", dpi=300)
            plt.close()
            plt.figure(figsize=(10,5))
            tcdiffdf.plot(ax=plt.gca(), title="Temporal Closeness over Time",xlabel="step", ylabel="mean difference")
            plt.savefig(outdir/"temporal-closeness-mean-differences.svg", dpi=300)
            plt.close()
            plt.figure(figsize=(10,5))
            tddiffdf.plot(ax=plt.gca(), title="Temporal Degree over Time",xlabel="step", ylabel="mean difference")
            plt.savefig(outdir/"temporal-degree-mean-differences.svg", dpi=300)
            plt.close()

            plt.figure(figsize=(10,5))
            tbvdiffdf.plot(ax=plt.gca(), title="Temporal Betweeness over Time",xlabel="step", ylabel="var difference")
            plt.savefig(outdir/"temporal-betweeness-var-differences.svg", dpi=300)
            plt.close()
            plt.figure(figsize=(10,5))
            tcvdiffdf.plot(ax=plt.gca(), title="Temporal Closeness over Time",xlabel="step", ylabel="var difference")
            plt.savefig(outdir/"temporal-closeness-var-differences.svg", dpi=300)
            plt.close()
            plt.figure(figsize=(10,5))
            tdvddifff.plot(ax=plt.gca(), title="Temporal Degree over Time",xlabel="step", ylabel="var difference")
            plt.savefig(outdir/"temporal-degree-var-differences.svg", dpi=300)
            plt.close()

        df = pandas.DataFrame.from_dict(log)
        df.to_csv(outdir/"log.csv")
        plt.figure(figsize=(10,5))
        df.plot(ax=plt.gca(), xlabel="step", ylabel="value", title=f"Loss over time")
        plt.savefig(outdir/"log.svg")
        plt.close()
        torch.save(generator.state_dict(), outdir/"latest-generator.ph")
        step += 1
    save_temporal_graph(generator, None, outdir, True)

def trainGraphGenTemporalOneShot(
    cfg: Dict,
    dataset="../generated/100k_frontend_graphs-temporal_unique.npy",
    outdir=Path("./output") / "LSTMAutoregressive",
    discriminator = GraphGenLSTMDescriminator(num_nodes, activation=torch.nn.Tanh, temporal=num_nodes-1, layers=[128]),
    generator = GraphGenGanGenerator(num_nodes, activation=torch.nn.Tanh, layers=[128])
):
    outdir = Path(outdir) / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(outdir, exist_ok=False)
    num_nodes = cfg["num_nodes"]
    time_steps = cfg["time_steps"]
    evaluate_size = cfg["evaluate_size"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    step = cfg["step"]
    clip=cfg["clip"]
    clip_value=cfg["clip_value"]

    dataset_tb = np.mean(np.load("generated/temporal_betweennesses_unique.npy"),axis=0)
    dataset_tc = np.mean(np.load("generated/temporal_closenesses_unique.npy"),axis=0)
    dataset_td = np.mean(np.load("generated/temporal_degrees_unique.npy"),axis=0)
    dataset_tbv = np.var(np.load("generated/temporal_betweennesses_unique.npy"),axis=0)
    dataset_tcv = np.var(np.load("generated/temporal_closenesses_unique.npy"),axis=0)
    dataset_tdv = np.var(np.load("generated/temporal_degrees_unique.npy"),axis=0)

    best_g_loss = inf
    t1 = time.perf_counter()
    loader = get_dataloader(dataset, batch_size)
    print(f"loaded data in {time.perf_counter() - t1:.2f}s")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    lr = cfg["lr"]
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    with open(outdir/"setup.txt", "w") as write_handle:
        write_handle.write(repr(generator))
        write_handle.write(repr(discriminator))
        write_handle.write(f"\nDataset: {dataset}\n")
        write_handle.write("\n".join([f"{key}: {item}" for key, item in cfg.items()]))
    log = {
        "d_generated": [],
        "d_real": [],
        "d_loss": [],
    }
    mean_differences = []
    tbs = []
    tds = []
    tcs = []
    tbvs = []
    tdvs = []
    tcvs = []
    tbsdiff = []
    tdsdiff = []
    tcsdiff = []
    tbvsdiff = []
    tdvsdiff = []
    tcvsdiff = []
    while step <= epochs:
        bar = tqdm(loader, desc=f"[{step}/{epochs}] train")
        for graph_batch in bar:
            # Train discriminator
            discriminator.train()
            generator.train()
            discriminator_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            random_data = torch.normal(0, 1, size=(batch_size, 1), device=device).float()
            generated: torch.Tensor = generator(random_data)
            d_generated = discriminator(generated.detach()).mean()
            d_real: torch.Tensor = discriminator(graph_batch).mean()
            d_loss: torch.Tensor = d_real - d_generated
            d_loss.backward()
            discriminator_optimizer.step()

            # Train generator
            discriminator_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            random_data = torch.normal(0, 1, size=(batch_size, 1), device=device)
            generated: torch.Tensor = generator(random_data)
            d_generated: torch.Tensor = discriminator(generated).mean()
            d_generated.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_value)
            generator_optimizer.step()
            log["d_generated"].append(d_generated.detach().cpu().item())
            log["d_loss"].append(d_loss.detach().cpu().item())
            log["d_real"].append(d_real.detach().cpu().item())
            bar.set_postfix({
                "d_generated": d_generated.detach().cpu().item(),
                "d_real": d_real.detach().cpu().item(),
                "d_loss": d_loss.detach().cpu().item(),
            })
            if d_generated.detach().cpu().item() < best_g_loss:
                best_g_loss = d_generated.detach().cpu().item()
                torch.save(generator.state_dict(), outdir/"best-loss-generator.ph")
                torch.save(discriminator.state_dict(), outdir/"best-loss-discriminator.ph")
        # Evaluate
        if step % 1 == 0: 
            with torch.no_grad():
                random_data = torch.normal(0, 1, size=(1000, 1), device=device)
                generated = generator(random_data) > 0.5
                generated = generated.reshape(-1, num_nodes, num_nodes, time_steps).cpu()
                tb, tc, td, tbv, tcv, tdv = get_temporal_metrics(generated, all=(step%20 == 0))
                tbs.append({nodes[i]: tb[i] for i in range(len(nodes))})
                tcs.append({nodes[i]: tc[i] for i in range(len(nodes))})
                tds.append({nodes[i]: td[i] for i in range(len(nodes))})
                tbvs.append({nodes[i]: tbv[i] for i in range(len(nodes))})
                tcvs.append({nodes[i]: tcv[i] for i in range(len(nodes))})
                tdvs.append({nodes[i]: tdv[i] for i in range(len(nodes))})
                tbdiff, tcdiff, tddiff = dataset_tb - tb, dataset_tc - tc, dataset_td - td
                tbvdiff, tcvdiff, tdvdiff = dataset_tbv - tbv, dataset_tcv - tcv, dataset_tdv - tdv
                tbsdiff.append({nodes[i]: tbdiff[i] for i in range(len(nodes))})
                tcsdiff.append({nodes[i]: tcdiff[i] for i in range(len(nodes))})
                tdsdiff.append({nodes[i]: tddiff[i] for i in range(len(nodes))})
                tbvsdiff.append({nodes[i]: tbvdiff[i] for i in range(len(nodes))})
                tcvsdiff.append({nodes[i]: tcvdiff[i] for i in range(len(nodes))})
                tdvsdiff.append({nodes[i]: tdvdiff[i] for i in range(len(nodes))})
                print(tbdiff, tcdiff, tddiff)
                print(tbvdiff, tcvdiff, tdvdiff)
                diffsum = abs(tbdiff) + abs(tcdiff) + abs(tddiff) + abs(tbvdiff) + abs(tcvdiff) + abs(tdvdiff)
                mean_differences.append(diffsum.sum())
            if diffsum.sum() == min(mean_differences):
                torch.save(generator.state_dict(), outdir/"best-generator.ph")
            tbdf = pandas.DataFrame.from_records(tbs, columns=nodes)
            tcdf = pandas.DataFrame.from_records(tcs, columns=nodes)
            tddf = pandas.DataFrame.from_records(tds, columns=nodes)

            tbdf.to_csv(outdir/"temporal-betweenesses-mean.csv")
            tcdf.to_csv(outdir/"temporal-clonsesses-mean.csv")
            tddf.to_csv(outdir/"temporal-degrees-mean.csv")

            tbdiffdf = pandas.DataFrame.from_records(tbsdiff, columns=nodes)
            tcdiffdf = pandas.DataFrame.from_records(tcsdiff, columns=nodes)
            tddiffdf = pandas.DataFrame.from_records(tdsdiff, columns=nodes)
            

            tbdiffdf.to_csv(outdir/"temporal-betweenesses-mean-differences.csv")
            tcdiffdf.to_csv(outdir/"temporal-clonsesses-mean-differences.csv")
            tddiffdf.to_csv(outdir/"temporal-degrees-mean-differences.csv")

            tbvdf = pandas.DataFrame.from_records(tbvs, columns=nodes)
            tcvdf = pandas.DataFrame.from_records(tcvs, columns=nodes)
            tdvdf = pandas.DataFrame.from_records(tdvs, columns=nodes)

            tbvdf.to_csv(outdir/"temporal-betweenesses-var.csv")
            tcvdf.to_csv(outdir/"temporal-clonsesses-var.csv")
            tdvdf.to_csv(outdir/"temporal-degrees-var.csv")

            tbvdiffdf = pandas.DataFrame.from_records(tbvsdiff, columns=nodes)
            tcvdiffdf = pandas.DataFrame.from_records(tcvsdiff, columns=nodes)
            tdvddifff = pandas.DataFrame.from_records(tdvsdiff, columns=nodes)
            

            tbvdiffdf.to_csv(outdir/"temporal-betweenesses-var-differences.csv")
            tcvdiffdf.to_csv(outdir/"temporal-clonsesses-var-differences.csv")
            tdvddifff.to_csv(outdir/"temporal-degrees-var-differences.csv")

            
            plt.figure(figsize=(10,5))
            tbdiffdf.plot(ax=plt.gca(), title="Temporal Betweeness over Time",xlabel="step", ylabel="mean difference")
            plt.savefig(outdir/"temporal-betweeness-mean-differences.svg", dpi=300)
            plt.close()
            plt.figure(figsize=(10,5))
            tcdiffdf.plot(ax=plt.gca(), title="Temporal Closeness over Time",xlabel="step", ylabel="mean difference")
            plt.savefig(outdir/"temporal-closeness-mean-differences.svg", dpi=300)
            plt.close()
            plt.figure(figsize=(10,5))
            tddiffdf.plot(ax=plt.gca(), title="Temporal Degree over Time",xlabel="step", ylabel="mean difference")
            plt.savefig(outdir/"temporal-degree-mean-differences.svg", dpi=300)
            plt.close()

            plt.figure(figsize=(10,5))
            tbvdiffdf.plot(ax=plt.gca(), title="Temporal Betweeness over Time",xlabel="step", ylabel="var difference")
            plt.savefig(outdir/"temporal-betweeness-var-differences.svg", dpi=300)
            plt.close()
            plt.figure(figsize=(10,5))
            tcvdiffdf.plot(ax=plt.gca(), title="Temporal Closeness over Time",xlabel="step", ylabel="var difference")
            plt.savefig(outdir/"temporal-closeness-var-differences.svg", dpi=300)
            plt.close()
            plt.figure(figsize=(10,5))
            tdvddifff.plot(ax=plt.gca(), title="Temporal Degree over Time",xlabel="step", ylabel="var difference")
            plt.savefig(outdir/"temporal-degree-var-differences.svg", dpi=300)
            plt.close()
        df = pandas.DataFrame.from_dict(log)
        df.to_csv(outdir/"log.csv")
        df.plot(ax=plt.gca(), xlabel="step", ylabel="value", title=f"Loss over time")
        plt.savefig(outdir/"log.svg")
        plt.close()
        torch.save(generator.state_dict(), outdir/"latest-generator.ph")
        torch.save(discriminator.state_dict(), outdir/"latest-discriminator.ph")
        step += 1
    save_temporal_graph(generator, discriminator, outdir, False)

def get_temporal_metrics(graphs: torch.Tensor, all=True):
    graphs = graphs.numpy()
    this_edges = set()
    this_nodes = set()
    tb = np.zeros((len(graphs), num_nodes))
    tc = np.zeros((len(graphs), num_nodes))
    td = np.zeros((len(graphs), num_nodes))
    for j in tqdm(range(graphs.shape[0])):
        el = graphs[j]
        tdi = overtime.TemporalDiGraph("Temporal Graph")
        this_edges = set()
        this_nodes = set()
        for t in range(el.shape[-1]):
            ones = np.transpose(np.nonzero(el[:,:,t]))
            for element in ones:
                element = tuple(element.tolist())
                if element[0] not in this_nodes:
                    this_nodes.add(element[0])
                    tdi.add_node(str(element[0]))
                if element[1] not in this_nodes:
                    this_nodes.add(element[1])
                    tdi.add_node(str(element[1]))
                if element not in this_edges:
                    this_edges.add(element)
                    tdi.add_edge(str(element[0]),str(element[1]), t+1, el.shape[-1]+2)
        try:
            for key, item in temporal_betweenness(tdi, optimality="foremost").items():
                tb[j, int(key) - 1] = item
        except Exception as e:
            print("Betweenness failed")
        try:
            for key, item in temporal_closeness(tdi, optimality="fastest").items():
                tc[j, int(key) - 1] = item
        except Exception as e:
            print("Closeness failed")
        try:
            for key, item in temporal_degree(tdi, in_out="out").items():
                td[j, int(key) - 1] = item
        except Exception as e:
            print("Degree failed")
    return np.mean(tb, axis=0), np.mean(tc, axis=0), np.mean(td, axis=0), np.var(tb, axis=0), np.var(tc, axis=0), np.var(td, axis=0)


def main():
    cfg = {
        "num_nodes":17,
        "time_steps":16,
        "evaluate_size": 1000,
        "epochs": 100,
        "batch_size": 1000,
        "step": 0,
        "clip": False,
        "clip_value": 1.0,
        "lr": 1e-4,
    }
    procs = []
    with ProcessPoolExecutor() as executor:
        procs.append(executor.submit(
            trainGraphGenTemporalSequence,
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "LSTMAutoregressiveSequence",
            GraphGenLSTMGeneratorNodeSoftmax(num_nodes, torch.nn.Tanh, 18),
        ))
        time.sleep(1)
        procs.append(executor.submit(
            trainGraphGenTemporalOneShot,
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "GANAutoregressive",
            GraphGenGanDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, [128]),
            GraphGenGanGenerator(num_nodes, torch.nn.Tanh, [128]),
        ))
        time.sleep(1)
        procs.append(executor.submit(
            trainGraphGenTemporalOneShot,
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "LinearOneShot",
            GraphGenLinearDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, [128]),
            GraphGenLinearGenerator(num_nodes, num_nodes-1, torch.nn.Tanh, [128]),
        ))
        time.sleep(1)
        procs.append(executor.submit(
            trainGraphGenTemporalOneShot,
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "LSTMAutoregressive",
            GraphGenLSTMDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, [128]),
            GraphGenLSTMGenerator(num_nodes, torch.nn.Tanh, [128]),
        ))
        time.sleep(1)
        cfg["clip"] = True
        trainGraphGenTemporalSequence(
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "LSTMAutoregressiveSequence",
            GraphGenLSTMGeneratorNodeSoftmax(num_nodes, torch.nn.Tanh, 18)
        )
        time.sleep(1)
        procs.append(executor.submit(
            trainGraphGenTemporalOneShot,
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "GANAutoregressive",
            GraphGenGanDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, [128]),
            GraphGenGanGenerator(num_nodes, torch.nn.Tanh, [128]),
        ))
        time.sleep(1)
        procs.append(executor.submit(
            trainGraphGenTemporalOneShot,
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "LinearOneShot",
            GraphGenLinearDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, [128]),
            GraphGenLinearGenerator(num_nodes, num_nodes-1, torch.nn.Tanh, [128]),
        ))
        time.sleep(1)
        procs.append(executor.submit(
            trainGraphGenTemporalOneShot,
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "LSTMAutoregressive",
            GraphGenLSTMDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, [128]),
            GraphGenLSTMGenerator(num_nodes, torch.nn.Tanh, [128]),
        ))
        time.sleep(1)
        procs.append(executor.submit(
            trainGraphGenTemporalOneShot,
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "LSTMAutoregressiveBetterDiscriminator",
            GraphGenLSTMDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, [128, 128]),
            GraphGenLSTMGenerator(num_nodes, torch.nn.Tanh, [128]),
        ))
        time.sleep(1)
        procs.append(executor.submit(
            trainGraphGenTemporalOneShot,
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "LSTMAutoregressiveBetterGenerator",
            GraphGenLSTMDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, []),
            GraphGenLSTMGenerator(num_nodes, torch.nn.Tanh, [128]),
        ))
        time.sleep(1)
        procs.append(executor.submit(
            trainGraphGenTemporalOneShot,
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "GANAutoregressiveStrongerGenerator",
            GraphGenGanDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, []),
            GraphGenGanGenerator(num_nodes, torch.nn.Tanh, [128]),
        ))
        time.sleep(1)
        procs.append(executor.submit(
            trainGraphGenTemporalOneShot,
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "GANAutoregressiveStrongerGenerator",
            GraphGenGanDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, [128]),
            GraphGenGanGenerator(num_nodes, torch.nn.Tanh, [128,128]),
        ))
        time.sleep(1)
        procs.append(executor.submit(
            trainGraphGenTemporalOneShot,
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "GANAutoregressiveStrongerDiscriminator",
            GraphGenGanDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, [128]),
            GraphGenGanGenerator(num_nodes, torch.nn.Tanh, []),
        ))
        time.sleep(1)
        procs.append(executor.submit(
            trainGraphGenTemporalOneShot,
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "GANAutoregressiveStrongerDiscriminator",
            GraphGenGanDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, [512,128,128]),
            GraphGenGanGenerator(num_nodes, torch.nn.Tanh, [128,512]),
        ))
        time.sleep(1)
        procs.append(executor.submit(
            trainGraphGenTemporalOneShot,
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "LinearOneShotStrongerGenerator",
            GraphGenLinearDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, []),
            GraphGenLinearGenerator(num_nodes, num_nodes-1, torch.nn.Tanh, [128]),
        ))
        time.sleep(1)
        procs.append(executor.submit(
            trainGraphGenTemporalOneShot,
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "LinearOneShotStrongerGenerator",
            GraphGenLinearDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, [128]),
            GraphGenLinearGenerator(num_nodes, num_nodes-1, torch.nn.Tanh, [128,128]),
        ))
        time.sleep(1)
        procs.append(executor.submit(
            trainGraphGenTemporalOneShot,
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "LinearOneShotStrongerDiscriminator",
            GraphGenLinearDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, [128]),
            GraphGenLinearGenerator(num_nodes, num_nodes-1, torch.nn.Tanh, []),
        ))
        time.sleep(1)
        procs.append(executor.submit(
            trainGraphGenTemporalOneShot,
            cfg,
            "generated/100k_frontend_graphs_temporal_unique.npy",
            Path("./output") / "LinearOneShotStrongerDiscriminator",
            GraphGenLinearDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, [512,128,128]),
            GraphGenLinearGenerator(num_nodes, num_nodes-1, torch.nn.Tanh, [128,512]),
        ))
    [proc.result() for proc in procs]

if __name__ == '__main__':
    main()
