import torch

from digital_drn import (
    BPTrainer,
    DRNGPTConfig,
    MQARConfig,
    MQARDataset,
    OptimizerConfig,
    SmallDRNGPT,
    TrainerConfig,
    build_dataloaders_from_config,
    build_model_from_config,
    build_trainer_from_config,
    generate_mqar_batch,
    load_experiment_config,
)


def test_mqar_batch_targets_only_query_positions():
    cfg = MQARConfig(
        num_pairs=3,
        num_queries=2,
        num_keys=8,
        num_values=5,
        seq_len=8,
        train_samples=4,
        eval_samples=4,
        ignore_index=-100,
    )
    generator = torch.Generator().manual_seed(0)
    inputs, targets = generate_mqar_batch(cfg, num_samples=4, generator=generator)

    pair_prefix_len = 2 * cfg.num_pairs
    query_positions = range(pair_prefix_len, pair_prefix_len + cfg.num_queries)

    assert inputs.shape == (4, cfg.resolved_seq_len)
    assert targets.shape == (4, cfg.resolved_seq_len)
    assert torch.all(targets[:, :pair_prefix_len] == cfg.ignore_index)
    assert torch.all(targets[:, pair_prefix_len + cfg.num_queries :] == cfg.ignore_index)

    for sample_inputs, sample_targets in zip(inputs, targets):
        key_to_value = {
            int(sample_inputs[pos].item()): int(sample_inputs[pos + 1].item())
            for pos in range(0, pair_prefix_len, 2)
        }
        for query_pos in query_positions:
            query_key = int(sample_inputs[query_pos].item())
            target_value = int(sample_targets[query_pos].item())
            assert target_value == key_to_value[query_key]
            assert cfg.value_offset <= target_value < cfg.value_offset + cfg.num_values


def test_mqar_dataset_is_deterministic_by_split_seed():
    cfg = MQARConfig(
        num_pairs=2,
        num_queries=2,
        num_keys=8,
        num_values=8,
        train_samples=6,
        eval_samples=4,
        seed=123,
    )

    train_a = MQARDataset(cfg, split="train")
    train_b = MQARDataset(cfg, split="train")
    eval_dataset = MQARDataset(cfg, split="eval")

    assert torch.equal(train_a.inputs, train_b.inputs)
    assert torch.equal(train_a.targets, train_b.targets)
    assert not torch.equal(train_a.inputs[: len(eval_dataset)], eval_dataset.inputs)


def test_mqar_config_builds_small_drn_gpt_and_dataloaders():
    cfg = load_experiment_config(config_name="mqar_drn_gpt_smoke")
    cfg["config"]["device"] = "cpu"
    cfg["config"]["save"] = False
    cfg["data"]["config"]["train_samples"] = 8
    cfg["data"]["config"]["eval_samples"] = 4
    cfg["data"]["config"]["batch_size"] = 4
    cfg["model"]["config"]["d_model"] = 16
    cfg["model"]["config"]["drn_hidden_dim"] = 16
    cfg["model"]["config"]["drn_num_iterations"] = 1
    cfg["trainer"]["epochs"] = 1
    cfg["trainer"]["log_every"] = 0
    cfg["trainer"]["eval_every"] = 1
    cfg["trainer"]["save_events"] = False
    cfg["trainer"]["train_num_iterations"] = 1
    cfg["trainer"]["eval_num_iterations"] = 1

    model = build_model_from_config(cfg)
    assert isinstance(model, SmallDRNGPT)
    assert model.config.vocab_size == cfg["data"]["config"]["vocab_size"]
    assert model.config.seq_len == cfg["data"]["config"]["seq_len"]
    assert model.blocks[0].mlp.block.energy._non_linearity == "perfect_diode"
    assert model.blocks[0].mlp.block.minimizer.mode == "asynchronous"

    train_loader, eval_loader = build_dataloaders_from_config(cfg)
    inputs, targets = next(iter(train_loader))
    assert inputs.shape == targets.shape == (4, cfg["data"]["config"]["seq_len"])
    assert int((targets != -100).sum().item()) == 4 * cfg["data"]["config"]["num_queries"]

    trainer = build_trainer_from_config(model, cfg)
    assert isinstance(trainer, BPTrainer)
    eval_metrics = trainer.evaluate(eval_loader, max_steps=1)
    assert eval_metrics.num_samples == 4 * cfg["data"]["config"]["num_queries"]
    trainer.close()


def test_bp_trainer_can_train_one_mqar_sequence_batch():
    torch.manual_seed(5)
    data_cfg = MQARConfig(
        num_pairs=2,
        num_queries=2,
        num_keys=12,
        num_values=12,
        train_samples=8,
        eval_samples=4,
        batch_size=4,
        seed=5,
    )
    train_dataset = MQARDataset(data_cfg, split="train")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False)

    model = SmallDRNGPT(
        DRNGPTConfig(
            vocab_size=data_cfg.resolved_vocab_size,
            seq_len=data_cfg.resolved_seq_len,
            d_model=12,
            n_heads=3,
            n_layers=1,
            mlp_ratio=1,
            dropout=0.0,
            drn_hidden_dim=12,
            drn_num_iterations=1,
            drn_non_linearity="perfect_diode",
            drn_weight_gains=0.1,
            drn_bias_gain=0.0,
        )
    )
    trainer = BPTrainer(
        model,
        TrainerConfig(
            epochs=1,
            optimizer=OptimizerConfig(name="sgd", lr=1.0e-2),
            device="cpu",
            seed=5,
            log_every=0,
            eval_every=0,
            save_best=False,
            save_events=False,
            train_num_iterations=1,
            eval_num_iterations=1,
        ),
    )
    metrics = trainer.train_epoch(train_loader, max_steps=1)

    assert metrics.num_samples == 4 * data_cfg.num_queries
    assert 0.0 <= metrics.accuracy <= 1.0
    assert model.blocks[0].mlp.block.energy.dense_weights[0].state.grad is not None
    trainer.close()
