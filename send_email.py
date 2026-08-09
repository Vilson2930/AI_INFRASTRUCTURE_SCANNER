# ============================================================
# send_email.py
# AI INFRASTRUCTURE SCANNER
# Institutional Opportunity Report Mailer
# ============================================================

import os
import smtplib
from pathlib import Path
from email.message import EmailMessage
from email.utils import formatdate


# ============================================================
# CONFIGURAÇÃO
# ============================================================

REPORTS_DIR = Path("reports")

PDF_FILE = (
    REPORTS_DIR
    / "relatorio_ai_infrastructure_scanner.pdf"
)


# ============================================================
# ANEXAR PDF
# ============================================================

def anexar_pdf(msg, caminho):

    caminho = Path(caminho)

    if not caminho.exists():

        raise FileNotFoundError(
            f"PDF não encontrado: {caminho}"
        )

    with open(caminho, "rb") as f:

        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="pdf",
            filename=caminho.name,
        )


# ============================================================
# ENVIO
# ============================================================

def main():

    smtp_server = os.getenv(
        "SMTP_SERVER"
    )

    smtp_port = int(
        os.getenv(
            "SMTP_PORT",
            "587",
        )
    )

    smtp_user = os.getenv(
        "SMTP_USER"
    )

    smtp_password = os.getenv(
        "SMTP_PASSWORD"
    )

    email_to = os.getenv(
        "EMAIL_TO"
    )

    # --------------------------------------------------------
    # VALIDAÇÃO
    # --------------------------------------------------------

    if not all(
        [
            smtp_server,
            smtp_user,
            smtp_password,
            email_to,
        ]
    ):

        raise Exception(
            "Configuração SMTP incompleta."
        )

    if not PDF_FILE.exists():

        raise FileNotFoundError(
            "Relatório institucional do "
            "AI Infrastructure Scanner não encontrado."
        )

    # --------------------------------------------------------
    # E-MAIL
    # --------------------------------------------------------

    msg = EmailMessage()

    msg["Subject"] = (
        "AI Infrastructure Scanner | "
        "Institutional Opportunity Report"
    )

    msg["From"] = smtp_user

    msg["To"] = email_to

    msg["Date"] = formatdate(
        localtime=True
    )

    corpo = """
Olá,

O ciclo de processamento do AI Infrastructure Scanner
foi concluído com sucesso.

O relatório institucional em PDF segue anexado.

Conteúdo do relatório:

• Resumo Executivo

• Entradas Fortes

• Entradas Aprovadas

• Pré-Entradas

• Ações aguardando volume

• Ações aguardando gatilho

• Ações aguardando pullback

• Ranking das Melhores Oportunidades

• Institutional Money Flow Score

• Technical Entry Score

• Entry Timing Score

• Relação Risco / Retorno

• Diagnóstico de Tendência

• Qualidade Estrutural da Tendência

• Risco de Extensão

• Metodologia

• Conclusão Executiva

---

Horizonte operacional:
Swing Trade de até 6 meses.

Relatório gerado automaticamente pelo GitHub Actions.

AI INFRASTRUCTURE SCANNER
Institutional Opportunity Report

---
"""

    msg.set_content(
        corpo
    )

    # --------------------------------------------------------
    # ANEXO
    # --------------------------------------------------------

    anexar_pdf(
        msg,
        PDF_FILE,
    )

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    print(
        "=" * 70
    )

    print(
        "ENVIANDO RELATÓRIO — AI INFRASTRUCTURE SCANNER"
    )

    print(
        "=" * 70
    )

    print(
        f"Destino : {email_to}"
    )

    print(
        f"Arquivo : {PDF_FILE.name}"
    )

    # --------------------------------------------------------
    # SMTP
    # --------------------------------------------------------

    with smtplib.SMTP(
        smtp_server,
        smtp_port,
    ) as server:

        server.starttls()

        server.login(
            smtp_user,
            smtp_password,
        )

        server.send_message(
            msg
        )

    print(
        "=" * 70
    )

    print(
        "RELATÓRIO ENVIADO COM SUCESSO"
    )

    print(
        "=" * 70
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()
