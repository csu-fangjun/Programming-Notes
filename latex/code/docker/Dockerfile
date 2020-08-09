
FROM ubuntu:18.04

RUN ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    apt-get update && \
    apt-get install -y \
      vim \
      make \
      cm-super \
      dvipng \
      ghostscript \
      make \
      latex-cjk-all \
      latex-cjk-common \
      latex-cjk-chinese \
      latexmk \
      lcdf-typetools \
      lmodern \
      poppler-utils \
      psutils \
      purifyeps \
      t1utils \
      tex-gyre \
      tex4ht \
      texlive-base \
      texlive-bibtex-extra \
      texlive-binaries \
      texlive-extra-utils \
      texlive-font-utils \
      texlive-fonts-extra \
      texlive-fonts-extra-links \
      texlive-fonts-recommended \
      texlive-formats-extra \
      texlive-lang-all \
      texlive-lang-chinese \
      texlive-lang-cjk \
      texlive-latex-base \
      texlive-latex-extra \
      texlive-latex-recommended \
      texlive-luatex \
      texlive-metapost \
      texlive-pictures \
      texlive-plain-generic \
      texlive-pstricks \
      texlive-publishers \
      texlive-science \
      texlive-xetex \
      texlive-bibtex-extra \
    && \
    rm -rf /var/lib/apt/lists/*

