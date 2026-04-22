import torch

from digital_drn import (
    BPTrainer,
    DRNGPTConfig,
    IGNORE_INDEX,
    MQARConfig,
    MQARDataset,
    OptimizerConfig,
    SmallDRNGPT,
    SmallGPTMQARDataset,
    TrainerConfig,
    build_dataloaders_from_config,
    build_model_from_config,
    build_trainer_from_config,
    generate_mqar_batch,
    generate_small_gpt_mqar,
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


def test_small_gpt_mqar_generation_matches_shifted_contract():
    vocab_size = 128
    seq_len = 32
    num_kv_pairs = 4
    inputs, labels = generate_small_gpt_mqar(
        vocab_size=vocab_size,
        num_examples=64,
        seq_len=seq_len,
        num_kv_pairs=num_kv_pairs,
        seed=456,
        random_non_queries=False,
    )

    assert inputs.shape == labels.shape == (64, seq_len)
    assert inputs.dtype == torch.long
    assert labels.dtype == torch.long

    supervised = labels != IGNORE_INDEX
    assert torch.all(supervised.sum(dim=1) == num_kv_pairs)
    assert torch.all(((supervised.nonzero(as_tuple=False)[:, 1] - (2 * num_kv_pairs)) % 2) == 0)

    for row in range(inputs.size(0)):
        prefix = {int(inputs[row, pos]): int(inputs[row, pos + 1]) for pos in range(0, 2 * num_kv_pairs, 2)}
        for query_pos in supervised[row].nonzero(as_tuple=False).flatten().tolist():
            assert prefix[int(inputs[row, query_pos])] == int(labels[row, query_pos])

    inputs_again, labels_again = generate_small_gpt_mqar(
        vocab_size=vocab_size,
        num_examples=64,
        seq_len=seq_len,
        num_kv_pairs=num_kv_pairs,
        seed=456,
        random_non_queries=False,
    )
    assert torch.equal(inputs, inputs_again)
    assert torch.equal(labels, labels_again)


def test_small_gpt_mqar_dataset_constructor_shape():
    dataset = SmallGPTMQARDataset(
        vocab_size=64,
        num_examples=8,
        seq_len=16,
        num_kv_pairs=2,
        seed=99,
    )

    assert len(dataset) == 8
    inputs, labels = dataset[0]
    assert inputs.shape == labels.shape == (16,)
    assert int(dataset.query_mask.sum().item()) == 8 * 2


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


def test_small_gpt_mqar_config_builds_full_benchmark_shape():
    cfg = load_experiment_config(config_name="mqar_small_gpt_drn")
    cfg["config"]["device"] = "cpu"
    cfg["config"]["save"] = False
    cfg["data"]["config"]["train_examples"] = 8
    cfg["data"]["config"]["val_examples"] = 4
    cfg["data"]["config"]["batch_size"] = 4
    cfg["model"]["config"]["d_model"] = 16
    cfg["model"]["config"]["n_heads"] = 4
    cfg["model"]["config"]["n_layers"] = 1
    cfg["model"]["config"]["mlp_ratio"] = 1
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
    assert model.config.vocab_size == 512
    assert model.config.seq_len == 64
    assert model.config.drn_signed_drive is True
    assert model.config.drn_non_linearity == "perfect_diode"

    train_loader, eval_loader = build_dataloaders_from_config(cfg)
    inputs, targets = next(iter(train_loader))
    assert inputs.shape == targets.shape == (4, 64)
    assert int((targets != IGNORE_INDEX).sum().item()) == 4 * cfg["data"]["config"]["num_kv_pairs"]

    trainer = build_trainer_from_config(model, cfg)
    eval_metrics = trainer.evaluate(eval_loader, max_steps=1)
    assert eval_metrics.num_samples == 4 * cfg["data"]["config"]["num_kv_pairs"]
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
