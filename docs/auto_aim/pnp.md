## 识别算法的像素浮动导致的 pnp 测距随机方差和距离的关系

假设是识别算法导致像素浮动长度不变，例如都是 2px。若物距 $u$，测量像素长度 $p$.

$$u = K p ^ {-1} $$

$$\dfrac{\mathrm{d}u}{\mathrm{d}p} = -K p ^ {-2}$$

$$\mathrm{d}u = K ^ {-1} u ^ 2 \mathrm{d}p$$

$$\mathrm{Var}(u) \propto u ^ 4 $$
