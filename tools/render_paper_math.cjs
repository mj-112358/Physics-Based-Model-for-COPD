"use strict";

const fs = require("fs");
const path = require("path");
const sharp = require("sharp");
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");

const root = path.resolve(__dirname, "..");
const outDir = path.join(root, "paper_assets");
fs.mkdirSync(outDir, { recursive: true });

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({ packages: AllPackages });
const svgOutput = new SVG({ fontCache: "local" });
const document = mathjax.document("", { InputJax: tex, OutputJax: svgOutput });

function renderSvg(latex) {
  const html = adaptor.outerHTML(document.convert(latex, { display: true }));
  const start = html.indexOf("<svg");
  const end = html.indexOf("</svg>");
  let svg = html.slice(start, end + 6).replace(/<\?xml[^>]*>/g, "");
  if (!svg.includes('xmlns="http://www.w3.org/2000/svg"')) {
    svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ');
  }
  return Buffer.from(svg.replace(/currentColor/g, "#000000"));
}

async function writeEquation(name, latex, width) {
  const target = path.join(outDir, name);
  await sharp(renderSvg(latex), { density: 420 })
    .resize({ width, fit: "inside", withoutEnlargement: false })
    .flatten({ background: "#ffffff" })
    .extend({ top: 42, bottom: 42, left: 60, right: 60, background: "#ffffff" })
    .png()
    .toFile(target);
}

async function main() {
await writeEquation(
  "model_equations.png",
  String.raw`\begin{aligned}
q(t;\boldsymbol{\theta})
&=\left(\frac{\alpha}{\tau_f}e^{-t/\tau_f}
+\frac{1-\alpha}{\tau_s}e^{-t/\tau_s}\right)
\left(1-e^{-t/\tau_o}\right),\\[4pt]
F(t;\boldsymbol{\theta})
&=\mathrm{FVC}\,
\frac{q(t;\boldsymbol{\theta})}
{\displaystyle\int_{0}^{T}q(u;\boldsymbol{\theta})\,du},
\qquad
V(t;\boldsymbol{\theta})=\int_{0}^{t}F(u;\boldsymbol{\theta})\,du,\\[4pt]
\widehat{\boldsymbol{\theta}}_{\tau}
&=\underset{\boldsymbol{\theta}\in\Theta}{\arg\min}
\sum_{k:\,t_k\le\tau}
\frac{\left[V_k-V(t_k;\boldsymbol{\theta})\right]^2}{\widehat{\sigma}^{2}},\\[4pt]
\widehat r_{\tau}
&=\frac{\mathrm{FEV}_1}{\widehat{\mathrm{FVC}}_{\tau}}.
\end{aligned}`,
  1850,
);

await writeEquation(
  "raw_decode_equations.png",
  String.raw`\begin{aligned}
Q_k&=0.1\,c\,d_k\quad [\mathrm{L\,s^{-1}}],\\[3pt]
V_k&=c\,\textstyle\sum_{j=1}^{k}d_j\quad [\mathrm{mL}].
\end{aligned}`,
  1150,
);

await writeEquation(
  "time_zero_equations.png",
  String.raw`\begin{aligned}
t_0&=t_p-\frac{V_p-V_o}{Q_p},\\[3pt]
\mathrm{FEV}_1&=V(t_0+1\,\mathrm{s})-V(t_0),\\[3pt]
\mathrm{FVC}_{\mathrm{direct}}&=V(t_{\mathrm{end}})-V(t_0).
\end{aligned}`,
  1450,
);

await writeEquation(
  "parameter_bounds.png",
  String.raw`\begin{aligned}
\boldsymbol{\theta}
&=(\tau_f,\tau_s,\alpha,\mathrm{FVC},\tau_o)\in\Theta,\\[3pt]
\Theta
&=[0.15,0.80]\,\mathrm{s}\times[1.00,5.00]\,\mathrm{s}
\times[0.25,0.85]\\
&\quad\times[2.00,6.00]\,\mathrm{L}\times[0.03,0.40]\,\mathrm{s}.
\end{aligned}`,
  1650,
);

await writeEquation(
  "metrics_equations.png",
  String.raw`\begin{aligned}
\mathrm{FNR}&=\frac{\mathrm{FN}}{\mathrm{TP}+\mathrm{FN}},
\qquad
\mathrm{specificity}=\frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}},\\[4pt]
\kappa&=\frac{p_o-p_e}{1-p_e}.
\end{aligned}`,
  1450,
);

await writeEquation(
  "analysis_equations.png",
  String.raw`\begin{aligned}
r_{\tau}
&=\frac{\mathrm{FEV}_1}{V(\tau)}
\ge \frac{\mathrm{FEV}_1}{\mathrm{FVC}}=r_{\infty},\\[4pt]
r_{\tau}-r_{\infty}
&=\frac{\mathrm{FEV}_1\,[\mathrm{FVC}-V(\tau)]}
{V(\tau)\,\mathrm{FVC}},\\[4pt]
J_{\tau}(\boldsymbol{\theta})
&=\frac{\partial V(t\le\tau;\boldsymbol{\theta})}
{\partial\boldsymbol{\theta}},
\qquad
I_{\tau}(\boldsymbol{\theta})=J_{\tau}^{\mathsf T}WJ_{\tau}.
\end{aligned}`,
  1650,
);

await writeEquation(
  "session_equations.png",
  String.raw`\begin{aligned}
\mathrm{FEV}_{1,i}^{\ast}&=\max_{j\in\mathcal A_i}\mathrm{FEV}_{1,ij},
&\mathrm{FVC}_{i}^{\ast}&=\max_{j\in\mathcal A_i}\mathrm{FVC}_{ij},\\[4pt]
r_i^{\ast}&=\frac{\mathrm{FEV}_{1,i}^{\ast}}{\mathrm{FVC}_{i}^{\ast}}.
\end{aligned}`,
  1500,
);

await writeEquation(
  "selective_equations.png",
  String.raw`\begin{aligned}
h&:\mathcal X\rightarrow\{0,1\},
&g&:\mathcal X\rightarrow\{0,1\},\\[4pt]
\operatorname{coverage}&=\Pr[g(X)=1],
&R_{\mathrm{sel}}
&=\frac{\mathbb E[\ell(h(X),Y)g(X)]}{\Pr[g(X)=1]}.
\end{aligned}`,
  1600,
);

await writeEquation(
  "complexity_equations.png",
  String.raw`\begin{aligned}
T_{\mathrm{residual}}(n)&=\mathcal O(n),
&T_{\mathrm{Jacobian}}(n,p)&=\mathcal O(pn),\\[4pt]
T_{\mathrm{fit}}(n,p,K)&=\mathcal O(Kpn),
&M_{\mathrm{fit}}(n)&=\mathcal O(n),
\qquad p=5.
\end{aligned}`,
  1550,
);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
