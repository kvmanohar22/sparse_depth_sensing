# Estimating Depth from RGB and Sparse Sensing

Paper Link: [arxiv](https://arxiv.org/abs/1804.02771)

# Requirements
- Ubuntu (Tested only on 16.04)
- Python 3
- Chainer
- ChainerCV
- cupy
- [imgplot](https://github.com/musyoku/imgplot/)

# TODO
- [x] Generate sparse inputs
- [ ] Build the network
- [ ] Implement training loop

# Current Progress

<div class="fig figcenter fighighlight">
  <img src="./tests/test_data/img.png" width="49%" style="margin-right:1px;">
  <img src="./tests/test_data/depth.png" width="50%">
</div>

## Downsampling factor of `24x24`

<div class="fig figcenter fighighlight">
  <img src="./tests/test_data/24_mask.png" width="32%" style="margin-right:1px;">
  <img src="./tests/test_data/24_s1.png" width="32%" style="margin-right:1px;">
  <img src="./tests/test_data/24_s2.png" width="34%">
</div>

## Downsampling factor of `48x48`

<div class="fig figcenter fighighlight">
  <img src="./tests/test_data/48_mask.png" width="32%" style="margin-right:1px;">
  <img src="./tests/test_data/48_s1.png" width="32%" style="margin-right:1px;">
  <img src="./tests/test_data/48_s2.png" width="34%">
</div>

