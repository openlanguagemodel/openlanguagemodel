
Architecture customisation is one of the biggest parts of the olm library. The library gives the freedom to create highly custom architectures in two ways; one through the traditional PyTorch method of creating custom classes and defining a forward pass; or through the block structure method with the components implemented in `olm.nn`.

--- 
<h3>Understanding the structure module</h3> 
The olm.nn.structure module provides multiple components which serve as containers and glue in the architectural structure of a language model. The module is made up of two major concepts, *Blocks* and *Combinators*. 

--- 

1. **Blocks**
	Blocks represent custom containers created by users to offer structure and organise flow of data. They can be used to wrap functionalities into repeatable, and reusable, blocks. The major feature of blocks is that they can be recursive; that is, you can have a block inside a block. This allows chaining of structures and makes writing code for the structure much easier. 

	 The block class inherits the `torch.nn.Module` takes in input, a list. This list can contain blocks,  elements implemented in `olm.nn`, or in general any `torch.nn.Module` type objects. When a forward function is called on a block object, the input is sequentially passed through each of the elements in the list, and then the output is finally returned. This is a parable to `torch.nn.Sequential` , however it is used at a much higher level with fewer lines of code to structure even custom-defined elements to create custom architectures. Let us take an example. 
	 
	 ```python
	 from olm.nn.structure import Block
	 
	 transformer_block = Block([
		 LayerNorm(...),
		 MultiHeadAttentionwithRoPE(...),
		 SwiGLUFFN(...),
		 MLP(...)
	 ])
	 
	 transformer_block.forward(input)
	 ```

	This is a simple example of how a Block can be used to chain operations and elements to create a language model structure. The real power of the block class comes in when chaining multiple blocks nested together.

	```python
	lm = Block([
		Embedding(...),
		transformer_block,
		OutputHead(...)
	])
	
	lm.forward(input)
	```

	This shows how easy and low verbosity it makes creating an architecture. For further information on the syntax check out the [[API Reference]].

--- 

2. **Combinators**
	Combinators are the keys to defining non-linear, and repeating structures. The olm library provides three basic combinators which cover the most commonly used structures, however it is trivial to implement new combinators by inheriting from the `BaseCombinator` class in `olm.nn.structure.combinators.base.py`. Let us cover the three combinators, and take an example to fully understand each.

	1. **Parallel** 
		The parallel combinator is used when an input is meant to be distributed individually across two (or more) channels. The computations are carried across each channel separately and sequentially, and then the outputs are put together with a specific function referred as merge. This function is an input to the Parallel object and must specify how the the outputs are put together, usually adding or concatenating. 

		`Args: 
			blocks: list containing blocks describing operations on each channel
			merge: function describing how outputs are put together 
			dim: dimension across which the combination should happen`


	2. **Repeat**
		The repeat combinator allows you to chain any number of blocks together in a row. This is the most used combinator as large language models often use many repeating blocks in a row together. module_func is an input which must be a function(usually a lambda function) which returns the block to be repeated. It is defined this way to make sure that different weights are generated each time.
	
		`Args:
			module_func: function which returns the block to be repeated 
			num_repeat: number of times the block must be repeated`


	3. **Residual**
		This combinator implements the commonly used method of residual connections used in computer vision. It takes a very simple input of the block for a residual connection to be put around. 
	
		`Args:
			block: block around which a residual connection to be put around`


Let us take a full example to understand each of the combinators, and blocks together.

~~~python

from olm.nn.structure.combinators import Residual, Parallel, Repeat

attention_block = Block([
	LayerNorm(...),
	MultiHeadAttentionwithRoPE(...)
])

feedforward = Block([
	LayerNorm(...),
	SwiGLUFFN(...)
])

  
transformer_block = Block([
	Residual(attention_block),
	Residual(feedforward)
])


large_mlp = Block([
	LayerNorm(...),
	SwiGLUFFN(...)
])


language_model = Block([
	Parallel([
		large_mlp(),
		Repeat(lambda: transformer_block(...), 32)
	], merge: torch.sum)

])

language_model.forward(input)

~~~

This example demonstrates the usage of all three combinators together. The transformer_block uses the Residual combinator, and applies a residual connection around attention_block and then sequentially feedforward. A language_model is then defined with two channels; one being the large_mlp, the other being 32 repeated transformer_blocks. After both channels are computed, they are merged together by adding both channels together. This shows how easy it can be to implement non-linear and repeating structures in the olm library. Check out the architectures implemented using the olm structure in the olm.nn.models submodule. 
    

