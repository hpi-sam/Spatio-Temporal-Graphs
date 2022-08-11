from datetime import datetime
from math import inf
import os
import time
from typing import Dict
import torch
from pathlib import Path
from generators.generation_gan_autoregressive import GraphGenGanDescriminator, GraphGenGanGenerator
from generators.generation_lstm_autoregressive import GraphGenLSTMGenerator, GraphGenLSTMDescriminator
from generators.generation_one_shot_linear import GraphGenLinearDescriminator, GraphGenLinearGenerator
from generators.utils import *
from tqdm import tqdm
import teneto
import pandas
import overtime
from overtime.algorithms import temporal_betweenness, temporal_degree, temporal_closeness, temporal_pagerank

num_nodes = 17
nodes = ['carts', 'carts-db', 'catalogue', 'catalogue-db', 'front-end', 'orders', 'orders-db', 'payment', 'queue-master', 'rabbitmq', 'session-db', 'shipping', 'user', 'user-db', 'worker1', 'worker2', 'master']

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

    best_g_loss = inf
    t1 = time.perf_counter()
    loader = get_dataloader(dataset, batch_size)
    print(f"loaded data in {time.perf_counter() - t1:.2f}s")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    lr = 1e-4
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=lr)
    generator_optimizer = torch.optim.SGD(generator.parameters(), lr=lr)
    with open(outdir/"setup.txt", "w") as write_handle:
        write_handle.write(repr(generator))
        write_handle.write(repr(discriminator))
        write_handle.write(f"\nBatch size: {batch_size}\n")
        write_handle.write(f"Learning rate: {lr}\n")
        write_handle.write(f"Dataset: {dataset}\n")
        write_handle.write(f"Grandient clipping: {clip}\n")
        write_handle.write(f"Grandient clipping value: {clip_value}\n")
    log = {
        "d_generated": [],
        "d_real": [],
        "d_loss": [],
    }
    mean_differences = []
    tbs = []
    tds = []
    tcs = []
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
        if step % 20 == 0: 
            with torch.no_grad():
                random_data = torch.normal(0, 1, size=(10, 1), device=device)
                generated = generator(random_data) > 0.5
                generated = generated.reshape(-1, num_nodes, num_nodes, time_steps).cpu()
                tb, tc, td = get_temporal_metrics(generated)
                tbs.append({nodes[i]: tb[i] for i in range(len(nodes))})
                tcs.append({nodes[i]: tc[i] for i in range(len(nodes))})
                tds.append({nodes[i]: td[i] for i in range(len(nodes))})
                tbdiff, tcdiff, tddiff = dataset_tb - tb, dataset_tc - tc, dataset_td - td
                print(tbdiff, tcdiff, tddiff)
                diffsum = abs(tbdiff) + abs(tcdiff) + abs(tddiff)
                mean_differences.append(diffsum.sum())
            if diffsum.sum() == min( mean_differences):
                torch.save(generator.state_dict(), outdir/"best-generator.ph")
                torch.save(discriminator.state_dict(), outdir/"best-discriminator.ph")
            tbdf = pandas.DataFrame.from_records(tbs, columns=nodes)
            tcdf = pandas.DataFrame.from_records(tcs, columns=nodes)
            tddf = pandas.DataFrame.from_records(tds, columns=nodes)
            tbdf.to_csv(outdir/"temporal-betweenesses.csv")
            tcdf.to_csv(outdir/"temporal-clonsesses.csv")
            tddf.to_csv(outdir/"temporal-degrees.csv")
            
            tbdf.plot(title="Temporal Betweeness over Time",xlabel="step", ylabel="difference")
            plt.savefig(outdir/"temporal-betweeness-differences.png")
            plt.close()
            tcdf.plot(title="Temporal Closeness over Time",xlabel="step", ylabel="difference")
            plt.savefig(outdir/"temporal-closeness-differences.png")
            plt.close()
            tddf.plot(title="Temporal Degree over Time",xlabel="step", ylabel="difference")
            plt.savefig(outdir/"temporal-degree-differences.png")
            plt.close()
        df = pandas.DataFrame.from_dict(log)
        df.to_csv(outdir/"log.csv")
        df.plot(xlabel="step", ylabel="value", title=f"Loss over time")
        plt.savefig(outdir/"log.png")
        plt.close()
        torch.save(generator.state_dict(), outdir/"latest-generator.ph")
        torch.save(discriminator.state_dict(), outdir/"latest-discriminator.ph")
        step += 1
    save_temporal_graph(generator, discriminator, outdir)

def get_temporal_metrics(graphs):
    tdi = overtime.TemporalDiGraph("Temporal Graph")
    this_edges = set()
    this_nodes = set()
    tb = np.zeros((len(graphs), num_nodes))
    tc = np.zeros((len(graphs), num_nodes))
    td = np.zeros((len(graphs), num_nodes))
    for index, graph in enumerate(tqdm(graphs)):
        for i in range(graph.size(-1)):
            ones = torch.nonzero(graph[:,:,i])
            for element in ones:
                element = tuple(element.numpy())
                if element[0] not in this_nodes:
                    this_nodes.add(element[0])
                    tdi.add_node(str(element[0]))
                if element[1] not in this_nodes:
                    this_nodes.add(element[1])
                    tdi.add_node(str(element[1]))
                if element not in this_edges:
                    this_edges.add(element)
                    tdi.add_edge(str(element[0]),str(element[1]), i+1, 17)
        try:
            for key, item in temporal_betweenness(tdi).items():
                tb[index, int(key) - 1] = item
        except Exception as e:
            pass
        try:
            for key, item in temporal_closeness(tdi).items():
                tc[index, int(key) - 1] = item
        except Exception as e:
            pass
        try:
            for key, item in temporal_degree(tdi).items():
                td[index, int(key) - 1] = item
        except Exception as e:
            pass
    return np.mean(tb, axis=0), np.mean(tc, axis=0), np.mean(td, axis=0)

def main():
    cfg = {
        "num_nodes":17,
        "time_steps":16,
        "evaluate_size": 1000,
        "epochs": 100,
        "batch_size": 1000,
        "step": 0,
        "clip": True,
        "clip_value": 1.0
    }
    # trainGraphGenTemporalOneShot(
    #     cfg,
    #     "generated/100k_frontend_graphs_temporal_unique.npy",
    #     Path("./output") / "GANAutoregressive",
    #     GraphGenGanDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, [128]),
    #     GraphGenGanGenerator(num_nodes, torch.nn.Tanh, [128]),
    # )
    trainGraphGenTemporalOneShot(
        cfg,
        "generated/100k_frontend_graphs_temporal_unique.npy",
        Path("./output") / "LinearOneShot",
        GraphGenLinearDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, [128]),
        GraphGenLinearGenerator(num_nodes, num_nodes-1, torch.nn.Tanh, [128]),
    )
    trainGraphGenTemporalOneShot(
        cfg,
        "generated/100k_frontend_graphs_temporal_unique.npy",
        Path("./output") / "LSTMAutoregressive",
        GraphGenLSTMDescriminator(num_nodes, num_nodes-1, torch.nn.Tanh, [128]),
        GraphGenLSTMGenerator(num_nodes, torch.nn.Tanh, [128]),
    )
    
if __name__ == '__main__':
    main()
