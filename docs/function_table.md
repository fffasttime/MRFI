# MRFI function table

## experiment

|Name|Description|Args|
|-|-|-|
|[`logspace_density`](../experiment/#mrfi.experiment.logspace_density)|Similar as np.logspace but the datapoint always contains $10^{-n}$|`low`,`high`,`density`,`return`|
|[`Acc_experiment`](../experiment/#mrfi.experiment.Acc_experiment)|Return classification accuracy on dataset.|`model`,`dataloader`,`return`|
|[`observeFI_experiment`](../experiment/#mrfi.experiment.observeFI_experiment)|Run fault inject experiment and return internal oberver results.|`fi_model`,`input_data`,`use_golden`,`observers_dict`,`return`|
|[`BER_Acc_experiment`](../experiment/#mrfi.experiment.BER_Acc_experiment)|Conduct a series of experimet under different bit error rate and get accuracy.|`fi_model`,`fi_selectors`,`dataloader`,`BER_range`,`bit_width`,`return`|
|[`BER_observe_experiment`](../experiment/#mrfi.experiment.BER_observe_experiment)|Conduct a series of experimet under different bit error rate and get all observers result.|`fi_model`,`fi_selectors`,`dataloader`,`BER_range`,`bit_width`,`use_golden`,`return`|
|[`Acc_golden`](../experiment/#mrfi.experiment.Acc_golden)|Evaluate model accuracy without error injection.|`fi_model`,`dataloader`,`disable_quantization`,`return`|
|[`observeFI_experiment_plus`](../experiment/#mrfi.experiment.observeFI_experiment_plus)|Run fault injection experiment and observe its internal effect.|`fi_model`,`input_data`,`method`,`pre_hook`,`kwargs`,`return`|
|[`get_activation_info`](../experiment/#mrfi.experiment.get_activation_info)|Observe model activations without fault injection.|`fi_model`,`input_data`,`method`,`pre_hook`,`kwargs`,`return`|
|[`get_weight_info`](../experiment/#mrfi.experiment.get_weight_info)|Observe model weights.|`model`,`method`,`weight_name`,`kwargs`,`return`|


## observer

|Name|Description|Args|
|-|-|-|
|[`BaseObserver`](../observer/#mrfi.observer.BaseObserver)|Basement of observers.||
|[`MinMax`](../observer/#mrfi.observer.MinMax)|:mag: Observe min/max range of tensors.||
|[`RMSE`](../observer/#mrfi.observer.RMSE)|:hammer_and_wrench: Root Mean Square Error metric between golden run and fault inject run.||
|[`SaveLast`](../observer/#mrfi.observer.SaveLast)|:mag: Simply save last inference internal tensor. ||
|[`MaxAbs`](../observer/#mrfi.observer.MaxAbs)|:mag: Observe max abs range of tensors.||
|[`MeanAbs`](../observer/#mrfi.observer.MeanAbs)|:mag: Mean of abs, a metric of scale of values||
|[`Std`](../observer/#mrfi.observer.Std)|:mag: Standard deviation of zero-mean values.||
|[`Shape`](../observer/#mrfi.observer.Shape)|:mag: Simply record tensor shape of last inference||
|[`MAE`](../observer/#mrfi.observer.MAE)|:hammer_and_wrench: Mean Absolute Error between golden run and fault inject run.||
|[`EqualRate`](../observer/#mrfi.observer.EqualRate)|:hammer_and_wrench: Compare how many value unchanged between golden run and fault inject run.||
|[`UniformSampling`](../observer/#mrfi.observer.UniformSampling)|:mag: Uniform sampling from tensors between all inference, up to 10000 samples.||


## selector

|Name|Description|Args|
|-|-|-|
|[`EmptySelector`](../selector/#mrfi.selector.EmptySelector)|No position selected by this, for debug or special use.||
|[`FixPosition`](../selector/#mrfi.selector.FixPosition)|Select fixed *one* position by coordinate.|`position`|
|[`FixPositions`](../selector/#mrfi.selector.FixPositions)|Select a list of fixed positions by coordinate.|`positions`|
|[`RandomPositionByNumber`](../selector/#mrfi.selector.RandomPositionByNumber)|Select n random positions.|`n`,`per_instance`|
|[`RandomPositionByRate`](../selector/#mrfi.selector.RandomPositionByRate)|Select random positions by rate.|`rate`,`poisson`|
|[`RandomPositionByRate_classic`](../selector/#mrfi.selector.RandomPositionByRate_classic)|Select random positions by rate.|`rate`|
|[`MaskedDimRandomPositionByNumber`](../selector/#mrfi.selector.MaskedDimRandomPositionByNumber)|Select n positions after specifed dimensions are masked.|`n`,`kwargs`|
|[`SelectedDimRandomPositionByNumber`](../selector/#mrfi.selector.SelectedDimRandomPositionByNumber)|Select n positions on selected coordinate.|`n`,`kwargs`|
|[`MaskedDimRandomPositionByRate`](../selector/#mrfi.selector.MaskedDimRandomPositionByRate)|Select by rate where some coordinate are masked.|`rate`,`poisson`,`kwargs`|
|[`SelectedDimRandomPositionByRate`](../selector/#mrfi.selector.SelectedDimRandomPositionByRate)|Select on some coordinate by rate.|`rate`,`poisson`,`kwargs`|
|[`FixedPixelByNumber`](../selector/#mrfi.selector.FixedPixelByNumber)|Select random channel on one fixed pixel (i.e. H * W dimension).|`n`,`pixel`,`per_instance`|


## error_mode

|Name|Description|Args|
|-|-|-|
|[`IntSignBitFlip`](../error_mode/#mrfi.error_mode.IntSignBitFlip)|Flip integer tensor on sign bit|`x_in`,`bit_width`|
|[`IntRandomBitFlip`](../error_mode/#mrfi.error_mode.IntRandomBitFlip)|Flip integer tensor on random bit|`x_in`,`bit_width`|
|[`IntFixedBitFlip`](../error_mode/#mrfi.error_mode.IntFixedBitFlip)|Flip integer tensor on specified bit|`x_in`,`bit_width`,`bit`|
|[`SetZero`](../error_mode/#mrfi.error_mode.SetZero)|Fault inject selected value to zero|`x`|
|[`SetValue`](../error_mode/#mrfi.error_mode.SetValue)|Fault inject value to specified value(s)|`x`,`value`|
|[`FloatRandomBitFlip`](../error_mode/#mrfi.error_mode.FloatRandomBitFlip)|Flip bit randomly on float values|`x_in`,`floattype`|
|[`FloatFixedBitFlip`](../error_mode/#mrfi.error_mode.FloatFixedBitFlip)|Flip specific bit on float values|`x_in`,`bit`,`floattype`|
|[`IntRandom`](../error_mode/#mrfi.error_mode.IntRandom)|Uniformly set to random integer in range.|`low`,`high`|
|[`IntRandomBit`](../error_mode/#mrfi.error_mode.IntRandomBit)|Set fault value to a random `bit_width`-bit integer.|`bit_width`,`signed`|
|[`UniformRandom`](../error_mode/#mrfi.error_mode.UniformRandom)|Uniform set to float number in range.|`low`,`high`|
|[`NormalRandom`](../error_mode/#mrfi.error_mode.NormalRandom)|Set fault injection value to a normal distribution.|`mean`,`std`|
|[`UniformDisturb`](../error_mode/#mrfi.error_mode.UniformDisturb)|Add uniform distribution noise to current value.|`low`,`high`|
|[`NormalDisturb`](../error_mode/#mrfi.error_mode.NormalDisturb)|Add normal distribution noise to current value.|`std`,`bias`|


## quantization

|Name|Description|Args|
|-|-|-|
|[`SymmericQuantization`](../quantization/#mrfi.quantization.SymmericQuantization)|Simple symmeric quantization.|`dynamic_range`,`scale_factor`|
|[`PositiveQuantization`](../quantization/#mrfi.quantization.PositiveQuantization)|Simple positive quantization.|`dynamic_range`,`scale_factor`|
|[`FixPointQuantization`](../quantization/#mrfi.quantization.FixPointQuantization)|Fixpoint quantization.|`decimal_bit`|


