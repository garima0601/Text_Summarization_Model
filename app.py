{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "18QuYdMlatLcJXYp8GDE79PkM0m2h5xSS",
      "authorship_tag": "ABX9TyMuvX/vAmbpjffx5xa/ZC/N",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/drishtinagpal/Fake_News_Article_Model/blob/main/Fake_News_Article_Model.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "maFxUOHHIkOB"
      },
      "outputs": [],
      "source": [
        "import pandas as pd"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "path=\"/content/drive/MyDrive/train_file.csv\""
      ],
      "metadata": {
        "id": "MSgjoCDnJIWn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df=pd.read_csv(path)"
      ],
      "metadata": {
        "id": "TFbLc70wJVjJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "8TEKMJ00Je0S",
        "outputId": "6ebb97cf-fd0e-4819-e9a3-78cec4601977"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   id                                              title              author  \\\n",
              "0   0  House Dem Aide: We Didn’t Even See Comey’s Let...       Darrell Lucus   \n",
              "1   1  FLYNN: Hillary Clinton, Big Woman on Campus - ...     Daniel J. Flynn   \n",
              "2   2                  Why the Truth Might Get You Fired  Consortiumnews.com   \n",
              "3   3  15 Civilians Killed In Single US Airstrike Hav...     Jessica Purkiss   \n",
              "4   4  Iranian woman jailed for fictional unpublished...      Howard Portnoy   \n",
              "\n",
              "                                                text  label  \n",
              "0  House Dem Aide: We Didn’t Even See Comey’s Let...      1  \n",
              "1  Ever get the feeling your life circles the rou...      0  \n",
              "2  Why the Truth Might Get You Fired October 29, ...      1  \n",
              "3  Videos 15 Civilians Killed In Single US Airstr...      1  \n",
              "4  Print \\nAn Iranian woman has been sentenced to...      1  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-38a8f9cc-679c-4a8d-a113-1b733daab9c3\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>id</th>\n",
              "      <th>title</th>\n",
              "      <th>author</th>\n",
              "      <th>text</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>\n",
              "      <td>Darrell Lucus</td>\n",
              "      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>FLYNN: Hillary Clinton, Big Woman on Campus - ...</td>\n",
              "      <td>Daniel J. Flynn</td>\n",
              "      <td>Ever get the feeling your life circles the rou...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>2</td>\n",
              "      <td>Why the Truth Might Get You Fired</td>\n",
              "      <td>Consortiumnews.com</td>\n",
              "      <td>Why the Truth Might Get You Fired October 29, ...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>3</td>\n",
              "      <td>15 Civilians Killed In Single US Airstrike Hav...</td>\n",
              "      <td>Jessica Purkiss</td>\n",
              "      <td>Videos 15 Civilians Killed In Single US Airstr...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4</td>\n",
              "      <td>Iranian woman jailed for fictional unpublished...</td>\n",
              "      <td>Howard Portnoy</td>\n",
              "      <td>Print \\nAn Iranian woman has been sentenced to...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-38a8f9cc-679c-4a8d-a113-1b733daab9c3')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-38a8f9cc-679c-4a8d-a113-1b733daab9c3 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-38a8f9cc-679c-4a8d-a113-1b733daab9c3');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-35176857-267c-4603-8117-aed4871e086a\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-35176857-267c-4603-8117-aed4871e086a')\"\n",
              "            title=\"Suggest charts.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "    background-color: #E8F0FE;\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: #1967D2;\n",
              "    height: 32px;\n",
              "    padding: 0 0 0 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: #E2EBFA;\n",
              "    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: #174EA6;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "    background-color: #3B4455;\n",
              "    fill: #D2E3FC;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart:hover {\n",
              "    background-color: #434B5C;\n",
              "    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "    fill: #FFFFFF;\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const charts = await google.colab.kernel.invokeFunction(\n",
              "          'suggestCharts', [key], {});\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-35176857-267c-4603-8117-aed4871e086a button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hazrKGLoLL3o",
        "outputId": "7f579d17-f781-4e4a-817f-02efa9d73a04"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 20800 entries, 0 to 20799\n",
            "Data columns (total 5 columns):\n",
            " #   Column  Non-Null Count  Dtype \n",
            "---  ------  --------------  ----- \n",
            " 0   id      20800 non-null  int64 \n",
            " 1   title   20242 non-null  object\n",
            " 2   author  18843 non-null  object\n",
            " 3   text    20761 non-null  object\n",
            " 4   label   20800 non-null  int64 \n",
            "dtypes: int64(2), object(3)\n",
            "memory usage: 812.6+ KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oTK_glfNLMCI",
        "outputId": "848d8371-d182-4d8d-e2ad-d995f4c573a4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "id           0\n",
              "title      558\n",
              "author    1957\n",
              "text        39\n",
              "label        0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df=df.drop(['id','title','author'],axis=1)"
      ],
      "metadata": {
        "id": "W8yzaRGULMPV"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "wxeh1_SkLtbE",
        "outputId": "7179e798-c0e3-40a8-b2ec-7e44624baf5c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                                text  label\n",
              "0  House Dem Aide: We Didn’t Even See Comey’s Let...      1\n",
              "1  Ever get the feeling your life circles the rou...      0\n",
              "2  Why the Truth Might Get You Fired October 29, ...      1\n",
              "3  Videos 15 Civilians Killed In Single US Airstr...      1\n",
              "4  Print \\nAn Iranian woman has been sentenced to...      1"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-ade3dd9c-7c8c-4eef-afd6-58abc779469c\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>text</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Ever get the feeling your life circles the rou...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Why the Truth Might Get You Fired October 29, ...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Videos 15 Civilians Killed In Single US Airstr...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Print \\nAn Iranian woman has been sentenced to...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-ade3dd9c-7c8c-4eef-afd6-58abc779469c')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-ade3dd9c-7c8c-4eef-afd6-58abc779469c button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-ade3dd9c-7c8c-4eef-afd6-58abc779469c');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-8f3074d5-f6df-4274-a727-bb12bf6d14ff\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-8f3074d5-f6df-4274-a727-bb12bf6d14ff')\"\n",
              "            title=\"Suggest charts.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "    background-color: #E8F0FE;\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: #1967D2;\n",
              "    height: 32px;\n",
              "    padding: 0 0 0 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: #E2EBFA;\n",
              "    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: #174EA6;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "    background-color: #3B4455;\n",
              "    fill: #D2E3FC;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart:hover {\n",
              "    background-color: #434B5C;\n",
              "    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "    fill: #FFFFFF;\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const charts = await google.colab.kernel.invokeFunction(\n",
              "          'suggestCharts', [key], {});\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-8f3074d5-f6df-4274-a727-bb12bf6d14ff button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install nltk"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xFTIUQwXLtlh",
        "outputId": "a1c3aaa7-6c4a-499d-b075-0a37d733e1d7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: nltk in /usr/local/lib/python3.10/dist-packages (3.8.1)\n",
            "Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from nltk) (8.1.7)\n",
            "Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from nltk) (1.3.2)\n",
            "Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.10/dist-packages (from nltk) (2023.6.3)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from nltk) (4.66.1)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import nltk"
      ],
      "metadata": {
        "id": "9pU4iLJoLts1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "nltk.download('all')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GR9dNrJ9McYF",
        "outputId": "32fc269a-ca1a-417c-90e2-99f276785af2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading collection 'all'\n",
            "[nltk_data]    | \n",
            "[nltk_data]    | Downloading package abc to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/abc.zip.\n",
            "[nltk_data]    | Downloading package alpino to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/alpino.zip.\n",
            "[nltk_data]    | Downloading package averaged_perceptron_tagger to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping taggers/averaged_perceptron_tagger.zip.\n",
            "[nltk_data]    | Downloading package averaged_perceptron_tagger_ru to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping\n",
            "[nltk_data]    |       taggers/averaged_perceptron_tagger_ru.zip.\n",
            "[nltk_data]    | Downloading package basque_grammars to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping grammars/basque_grammars.zip.\n",
            "[nltk_data]    | Downloading package bcp47 to /root/nltk_data...\n",
            "[nltk_data]    | Downloading package biocreative_ppi to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/biocreative_ppi.zip.\n",
            "[nltk_data]    | Downloading package bllip_wsj_no_aux to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping models/bllip_wsj_no_aux.zip.\n",
            "[nltk_data]    | Downloading package book_grammars to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping grammars/book_grammars.zip.\n",
            "[nltk_data]    | Downloading package brown to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/brown.zip.\n",
            "[nltk_data]    | Downloading package brown_tei to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/brown_tei.zip.\n",
            "[nltk_data]    | Downloading package cess_cat to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/cess_cat.zip.\n",
            "[nltk_data]    | Downloading package cess_esp to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/cess_esp.zip.\n",
            "[nltk_data]    | Downloading package chat80 to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/chat80.zip.\n",
            "[nltk_data]    | Downloading package city_database to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/city_database.zip.\n",
            "[nltk_data]    | Downloading package cmudict to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/cmudict.zip.\n",
            "[nltk_data]    | Downloading package comparative_sentences to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/comparative_sentences.zip.\n",
            "[nltk_data]    | Downloading package comtrans to /root/nltk_data...\n",
            "[nltk_data]    | Downloading package conll2000 to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/conll2000.zip.\n",
            "[nltk_data]    | Downloading package conll2002 to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/conll2002.zip.\n",
            "[nltk_data]    | Downloading package conll2007 to /root/nltk_data...\n",
            "[nltk_data]    | Downloading package crubadan to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/crubadan.zip.\n",
            "[nltk_data]    | Downloading package dependency_treebank to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/dependency_treebank.zip.\n",
            "[nltk_data]    | Downloading package dolch to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/dolch.zip.\n",
            "[nltk_data]    | Downloading package europarl_raw to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/europarl_raw.zip.\n",
            "[nltk_data]    | Downloading package extended_omw to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    | Downloading package floresta to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/floresta.zip.\n",
            "[nltk_data]    | Downloading package framenet_v15 to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/framenet_v15.zip.\n",
            "[nltk_data]    | Downloading package framenet_v17 to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/framenet_v17.zip.\n",
            "[nltk_data]    | Downloading package gazetteers to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/gazetteers.zip.\n",
            "[nltk_data]    | Downloading package genesis to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/genesis.zip.\n",
            "[nltk_data]    | Downloading package gutenberg to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/gutenberg.zip.\n",
            "[nltk_data]    | Downloading package ieer to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/ieer.zip.\n",
            "[nltk_data]    | Downloading package inaugural to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/inaugural.zip.\n",
            "[nltk_data]    | Downloading package indian to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/indian.zip.\n",
            "[nltk_data]    | Downloading package jeita to /root/nltk_data...\n",
            "[nltk_data]    | Downloading package kimmo to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/kimmo.zip.\n",
            "[nltk_data]    | Downloading package knbc to /root/nltk_data...\n",
            "[nltk_data]    | Downloading package large_grammars to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping grammars/large_grammars.zip.\n",
            "[nltk_data]    | Downloading package lin_thesaurus to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/lin_thesaurus.zip.\n",
            "[nltk_data]    | Downloading package mac_morpho to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/mac_morpho.zip.\n",
            "[nltk_data]    | Downloading package machado to /root/nltk_data...\n",
            "[nltk_data]    | Downloading package masc_tagged to /root/nltk_data...\n",
            "[nltk_data]    | Downloading package maxent_ne_chunker to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping chunkers/maxent_ne_chunker.zip.\n",
            "[nltk_data]    | Downloading package maxent_treebank_pos_tagger to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping taggers/maxent_treebank_pos_tagger.zip.\n",
            "[nltk_data]    | Downloading package moses_sample to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping models/moses_sample.zip.\n",
            "[nltk_data]    | Downloading package movie_reviews to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/movie_reviews.zip.\n",
            "[nltk_data]    | Downloading package mte_teip5 to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/mte_teip5.zip.\n",
            "[nltk_data]    | Downloading package mwa_ppdb to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping misc/mwa_ppdb.zip.\n",
            "[nltk_data]    | Downloading package names to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/names.zip.\n",
            "[nltk_data]    | Downloading package nombank.1.0 to /root/nltk_data...\n",
            "[nltk_data]    | Downloading package nonbreaking_prefixes to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/nonbreaking_prefixes.zip.\n",
            "[nltk_data]    | Downloading package nps_chat to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/nps_chat.zip.\n",
            "[nltk_data]    | Downloading package omw to /root/nltk_data...\n",
            "[nltk_data]    | Downloading package omw-1.4 to /root/nltk_data...\n",
            "[nltk_data]    | Downloading package opinion_lexicon to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/opinion_lexicon.zip.\n",
            "[nltk_data]    | Downloading package panlex_swadesh to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    | Downloading package paradigms to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/paradigms.zip.\n",
            "[nltk_data]    | Downloading package pe08 to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/pe08.zip.\n",
            "[nltk_data]    | Downloading package perluniprops to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping misc/perluniprops.zip.\n",
            "[nltk_data]    | Downloading package pil to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/pil.zip.\n",
            "[nltk_data]    | Downloading package pl196x to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/pl196x.zip.\n",
            "[nltk_data]    | Downloading package porter_test to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping stemmers/porter_test.zip.\n",
            "[nltk_data]    | Downloading package ppattach to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/ppattach.zip.\n",
            "[nltk_data]    | Downloading package problem_reports to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/problem_reports.zip.\n",
            "[nltk_data]    | Downloading package product_reviews_1 to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/product_reviews_1.zip.\n",
            "[nltk_data]    | Downloading package product_reviews_2 to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/product_reviews_2.zip.\n",
            "[nltk_data]    | Downloading package propbank to /root/nltk_data...\n",
            "[nltk_data]    | Downloading package pros_cons to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/pros_cons.zip.\n",
            "[nltk_data]    | Downloading package ptb to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/ptb.zip.\n",
            "[nltk_data]    | Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping tokenizers/punkt.zip.\n",
            "[nltk_data]    | Downloading package qc to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/qc.zip.\n",
            "[nltk_data]    | Downloading package reuters to /root/nltk_data...\n",
            "[nltk_data]    | Downloading package rslp to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping stemmers/rslp.zip.\n",
            "[nltk_data]    | Downloading package rte to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/rte.zip.\n",
            "[nltk_data]    | Downloading package sample_grammars to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping grammars/sample_grammars.zip.\n",
            "[nltk_data]    | Downloading package semcor to /root/nltk_data...\n",
            "[nltk_data]    | Downloading package senseval to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/senseval.zip.\n",
            "[nltk_data]    | Downloading package sentence_polarity to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/sentence_polarity.zip.\n",
            "[nltk_data]    | Downloading package sentiwordnet to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/sentiwordnet.zip.\n",
            "[nltk_data]    | Downloading package shakespeare to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/shakespeare.zip.\n",
            "[nltk_data]    | Downloading package sinica_treebank to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/sinica_treebank.zip.\n",
            "[nltk_data]    | Downloading package smultron to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/smultron.zip.\n",
            "[nltk_data]    | Downloading package snowball_data to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    | Downloading package spanish_grammars to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping grammars/spanish_grammars.zip.\n",
            "[nltk_data]    | Downloading package state_union to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/state_union.zip.\n",
            "[nltk_data]    | Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/stopwords.zip.\n",
            "[nltk_data]    | Downloading package subjectivity to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/subjectivity.zip.\n",
            "[nltk_data]    | Downloading package swadesh to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/swadesh.zip.\n",
            "[nltk_data]    | Downloading package switchboard to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/switchboard.zip.\n",
            "[nltk_data]    | Downloading package tagsets to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping help/tagsets.zip.\n",
            "[nltk_data]    | Downloading package timit to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/timit.zip.\n",
            "[nltk_data]    | Downloading package toolbox to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/toolbox.zip.\n",
            "[nltk_data]    | Downloading package treebank to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/treebank.zip.\n",
            "[nltk_data]    | Downloading package twitter_samples to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/twitter_samples.zip.\n",
            "[nltk_data]    | Downloading package udhr to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/udhr.zip.\n",
            "[nltk_data]    | Downloading package udhr2 to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/udhr2.zip.\n",
            "[nltk_data]    | Downloading package unicode_samples to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/unicode_samples.zip.\n",
            "[nltk_data]    | Downloading package universal_tagset to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping taggers/universal_tagset.zip.\n",
            "[nltk_data]    | Downloading package universal_treebanks_v20 to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    | Downloading package vader_lexicon to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    | Downloading package verbnet to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/verbnet.zip.\n",
            "[nltk_data]    | Downloading package verbnet3 to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/verbnet3.zip.\n",
            "[nltk_data]    | Downloading package webtext to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/webtext.zip.\n",
            "[nltk_data]    | Downloading package wmt15_eval to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping models/wmt15_eval.zip.\n",
            "[nltk_data]    | Downloading package word2vec_sample to\n",
            "[nltk_data]    |     /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping models/word2vec_sample.zip.\n",
            "[nltk_data]    | Downloading package wordnet to /root/nltk_data...\n",
            "[nltk_data]    | Downloading package wordnet2021 to /root/nltk_data...\n",
            "[nltk_data]    | Downloading package wordnet2022 to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/wordnet2022.zip.\n",
            "[nltk_data]    | Downloading package wordnet31 to /root/nltk_data...\n",
            "[nltk_data]    | Downloading package wordnet_ic to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/wordnet_ic.zip.\n",
            "[nltk_data]    | Downloading package words to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/words.zip.\n",
            "[nltk_data]    | Downloading package ycoe to /root/nltk_data...\n",
            "[nltk_data]    |   Unzipping corpora/ycoe.zip.\n",
            "[nltk_data]    | \n",
            "[nltk_data]  Done downloading collection all\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "True"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "nltk.download(\"stopwords\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xrW8brl2Mch_",
        "outputId": "53ff4fe1-d5b8-4dea-8e5d-ac76531d1abb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "True"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from nltk.corpus import stopwords"
      ],
      "metadata": {
        "id": "Xcit2c4BMcsy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import re"
      ],
      "metadata": {
        "id": "vkjsmj6_NDKh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from nltk.stem.porter import PorterStemmer"
      ],
      "metadata": {
        "id": "3cTx-TBFNDRy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "port_stem=PorterStemmer()"
      ],
      "metadata": {
        "id": "4Rw1IzkWNZuh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def stemming(content):\n",
        "    con=re.sub('[^a-zA-Z]',' ',str(content))\n",
        "    con=con.lower()\n",
        "    con=con.split()\n",
        "    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]\n",
        "    con=' '.join(con)\n",
        "    return con"
      ],
      "metadata": {
        "id": "buPI8CmJNaMO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "stemming('my Name is Drishti000')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "id": "_NmVkLYNPD9t",
        "outputId": "cf46afa0-0ba7-4c1c-91f5-2e993d7dbc58"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'name drishti'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df['text']=df['text'].apply(stemming)"
      ],
      "metadata": {
        "id": "6Dj3kgugPEI9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "0C7TeN9sXczq",
        "outputId": "1718698a-2e8e-483d-edcc-a109bfc26dbd"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                                text  label\n",
              "0  hous dem aid even see comey letter jason chaff...      1\n",
              "1  ever get feel life circl roundabout rather hea...      0\n",
              "2  truth might get fire octob tension intellig an...      1\n",
              "3  video civilian kill singl us airstrik identifi...      1\n",
              "4  print iranian woman sentenc six year prison ir...      1"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-e0511abb-8e56-4876-a9ff-a6bfaa588892\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>text</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>hous dem aid even see comey letter jason chaff...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>ever get feel life circl roundabout rather hea...</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>truth might get fire octob tension intellig an...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>video civilian kill singl us airstrik identifi...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>print iranian woman sentenc six year prison ir...</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-e0511abb-8e56-4876-a9ff-a6bfaa588892')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-e0511abb-8e56-4876-a9ff-a6bfaa588892 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-e0511abb-8e56-4876-a9ff-a6bfaa588892');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-40dc2824-e914-44c7-a1ab-413161f3e389\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-40dc2824-e914-44c7-a1ab-413161f3e389')\"\n",
              "            title=\"Suggest charts.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "    background-color: #E8F0FE;\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: #1967D2;\n",
              "    height: 32px;\n",
              "    padding: 0 0 0 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: #E2EBFA;\n",
              "    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: #174EA6;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "    background-color: #3B4455;\n",
              "    fill: #D2E3FC;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart:hover {\n",
              "    background-color: #434B5C;\n",
              "    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "    fill: #FFFFFF;\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const charts = await google.colab.kernel.invokeFunction(\n",
              "          'suggestCharts', [key], {});\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-40dc2824-e914-44c7-a1ab-413161f3e389 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 26
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install scikit-learn"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "L3pOKlxDhFOT",
        "outputId": "509060f0-9088-4b3c-8f2b-889403725393"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (1.2.2)\n",
            "Requirement already satisfied: numpy>=1.17.3 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.23.5)\n",
            "Requirement already satisfied: scipy>=1.3.2 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.10.1)\n",
            "Requirement already satisfied: joblib>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.3.2)\n",
            "Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (3.2.0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split"
      ],
      "metadata": {
        "id": "lUcDbtsrhFhj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x=df['text']"
      ],
      "metadata": {
        "id": "54gAuPxbhFwU"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y=df['label']"
      ],
      "metadata": {
        "id": "fkNi9-WfhF-f"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_train ,x_test, y_train ,y_test=train_test_split(x,y,test_size=0.25)"
      ],
      "metadata": {
        "id": "v-DeeKHghGI3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.feature_extraction.text import TfidfVectorizer"
      ],
      "metadata": {
        "id": "u22ildKlhGQv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "vect=TfidfVectorizer()"
      ],
      "metadata": {
        "id": "8c7s6ySchGW6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_train=vect.fit_transform(x_train)"
      ],
      "metadata": {
        "id": "9yifnxvJhGb9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x_test=vect.transform(x_test)"
      ],
      "metadata": {
        "id": "y7YZAMwNiMfl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.tree import DecisionTreeClassifier"
      ],
      "metadata": {
        "id": "q73axWAsiMv7"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model=DecisionTreeClassifier()"
      ],
      "metadata": {
        "id": "LNA5IpBxireO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.fit(x_train,y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 75
        },
        "id": "PJUfsBRgirtb",
        "outputId": "7922da38-7f4f-43c0-c4f9-b70b11b7f8ec"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DecisionTreeClassifier()"
            ],
            "text/html": [
              "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>DecisionTreeClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier()</pre></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 39
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "prediction=model.predict(x_test)"
      ],
      "metadata": {
        "id": "qKGOkdR7ir41"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "prediction"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "z4Sc0lCnisBa",
        "outputId": "749aa6d3-0d9a-4cf6-a63b-13838bcf5330"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0, 1, 0, ..., 0, 1, 1])"
            ]
          },
          "metadata": {},
          "execution_count": 41
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.score(x_test,y_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aMOkaHzQisKQ",
        "outputId": "349482d1-6f82-41a6-997f-54566793e4d9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8701923076923077"
            ]
          },
          "metadata": {},
          "execution_count": 42
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pickle"
      ],
      "metadata": {
        "id": "selFdXtjisRw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pickle.dump(vect,open('vector.pkl','wb'))"
      ],
      "metadata": {
        "id": "ijhm9hFwisYi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pickle.dump(model,open('model.pkl','wb'))"
      ],
      "metadata": {
        "id": "IaEjBSogisfB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "vector_form=pickle.load(open('vector.pkl','rb'))"
      ],
      "metadata": {
        "id": "QdXcKGGVjcpf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "load_model=pickle.load(open('model.pkl','rb'))"
      ],
      "metadata": {
        "id": "5uvoeGGkjc6s"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def fakenews(news):\n",
        "    news=stemming(news)\n",
        "    input_data=[news]\n",
        "    vector_form1=vector_form.transform(input_data)\n",
        "    prediction=load_model.predict(vector_form1)\n",
        "    return prediction"
      ],
      "metadata": {
        "id": "KoNo9phWjdAq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "val=fakenews(\"\")"
      ],
      "metadata": {
        "id": "zIValFGXjq3X"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "if val==[0]:\n",
        "    print('Reliable')\n",
        "else:\n",
        "    print('Unreliable')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DoCZHRq3jrIY",
        "outputId": "2a2fcc47-0dc0-477d-f397-e3d176341167"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Unreliable\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "bmiew9espa96"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
