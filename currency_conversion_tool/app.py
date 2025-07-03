from turtle import color
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import requests
import re
import json

load_dotenv()

st.title("💸Currency Conversion Tool")

currencies = [
    "Algerian Dinar (DZD)", "Angolan Kwanza (AOA)", "Botswana Pula (BWP)", "Burundi Franc (BIF)",
    "Cape Verde Escudo (CVE)", "Central African CFA Franc (XAF)", "Comorian Franc (KMF)", 
    "Congolese Franc (CDF)", "Egyptian Pound (EGP)", "Eritrean Nakfa (ERN)", "Ethiopian Birr (ETB)",
    "Gambian Dalasi (GMD)", "Ghanaian Cedi (GHS)", "Guinean Franc (GNF)", "Kenyan Shilling (KES)",
    "Lesotho Loti (LSL)", "Liberian Dollar (LRD)", "Libyan Dinar (LYD)", "Malagasy Ariary (MGA)",
    "Malawian Kwacha (MWK)", "Mauritanian Ouguiya (MRU)", "Mauritian Rupee (MUR)", "Moroccan Dirham (MAD)",
    "Mozambican Metical (MZN)", "Namibian Dollar (NAD)", "Nigerian Naira (NGN)", "Rwandan Franc (RWF)",
    "Sao Tome & Principe Dobra (STN)", "Seychellois Rupee (SCR)", "Sierra Leonean Leone (SLL)",
    "Somali Shilling (SOS)", "South African Rand (ZAR)", "South Sudanese Pound (SSP)", "Sudanese Pound (SDG)",
    "Tanzanian Shilling (TZS)", "Tunisian Dinar (TND)", "Ugandan Shilling (UGX)", "Zambian Kwacha (ZMW)",
    "Afghan Afghani (AFN)", "Armenian Dram (AMD)", "Azerbaijani Manat (AZN)", "Bahraini Dinar (BHD)",
    "Bangladeshi Taka (BDT)", "Bhutanese Ngultrum (BTN)", "Brunei Dollar (BND)", "Cambodian Riel (KHR)",
    "Chinese Yuan (CNY)", "Cypriot Euro (EUR)", "Georgian Lari (GEL)", "Hong Kong Dollar (HKD)",
    "Indian Rupee (INR)", "Indonesian Rupiah (IDR)", "Iranian Rial (IRR)", "Iraqi Dinar (IQD)",
    "Israeli New Shekel (ILS)", "Japanese Yen (JPY)", "Jordanian Dinar (JOD)", "Kazakhstani Tenge (KZT)",
    "Kuwaiti Dinar (KWD)", "Kyrgyzstani Som (KGS)", "Lao Kip (LAK)", "Lebanese Pound (LBP)",
    "Malaysian Ringgit (MYR)", "Maldivian Rufiyaa (MVR)", "Mongolian Tögrög (MNT)", "Myanmar Kyat (MMK)",
    "Nepalese Rupee (NPR)", "North Korean Won (KPW)", "Omani Rial (OMR)", "Pakistani Rupee (PKR)",
    "Philippine Peso (PHP)", "Qatari Riyal (QAR)", "Saudi Riyal (SAR)", "Singapore Dollar (SGD)",
    "South Korean Won (KRW)", "Sri Lankan Rupee (LKR)", "Syrian Pound (SYP)", "Tajikistani Somoni (TJS)",
    "Thai Baht (THB)", "Turkish Lira (TRY)", "Turkmenistani Manat (TMT)", "UAE Dirham (AED)",
    "Uzbekistani Som (UZS)", "Vietnamese Dong (VND)", "Yemeni Rial (YER)",
    "Albanian Lek (ALL)", "Andorran Euro (EUR)", "Belarusian Ruble (BYN)",
    "Bosnia-Herzegovina Convertible Mark (BAM)", "British Pound Sterling (GBP)", "Bulgarian Lev (BGN)",
    "Croatian Kuna (HRK)", "Czech Koruna (CZK)", "Danish Krone (DKK)", "Euro (EUR)",
    "Icelandic Króna (ISK)", "Macedonian Denar (MKD)", "Moldovan Leu (MDL)", "Norwegian Krone (NOK)",
    "Polish Złoty (PLN)", "Romanian Leu (RON)", "Russian Ruble (RUB)", "Serbian Dinar (RSD)",
    "Swiss Franc (CHF)", "Ukrainian Hryvnia (UAH)",
    "Argentine Peso (ARS)", "Barbadian Dollar (BBD)", "Belize Dollar (BZD)", "Boliviano (BOB)",
    "Brazilian Real (BRL)", "Canadian Dollar (CAD)", "Chilean Peso (CLP)", "Colombian Peso (COP)",
    "Costa Rican Colón (CRC)", "Cuban Peso (CUP)", "Dominican Peso (DOP)", "East Caribbean Dollar (XCD)",
    "Ecuadorian Dollar (USD)", "El Salvadoran Colón (SVC)", "Guatemalan Quetzal (GTQ)", "Guyanaese Dollar (GYD)",
    "Haitian Gourde (HTG)", "Honduran Lempira (HNL)", "Jamaican Dollar (JMD)", "Mexican Peso (MXN)",
    "Nicaraguan Córdoba (NIO)", "Panamanian Balboa (PAB)", "Paraguayan Guaraní (PYG)", "Peruvian Sol (PEN)",
    "Trinidad & Tobago Dollar (TTD)", "United States Dollar (USD)", "Uruguayan Peso (UYU)", "Venezuelan Bolívar (VES)",
    "Australian Dollar (AUD)", "Fijian Dollar (FJD)", "New Zealand Dollar (NZD)", 
    "Papua New Guinean Kina (PGK)", "Samoan Tala (WST)", "Solomon Islands Dollar (SBD)",
    "Tongan Paʻanga (TOP)", "Tuvaluan Dollar (AUD)", "Vanuatu Vatu (VUV)"
]


llm = ChatGroq(model="llama-3.1-8b-instant")

@tool
def get_conversion_rate(base_currency: str, target_currency: str) -> float:
    """
    This function fetches the currency conversion factor between a given base currency and a target currency
    """
    url = f'https://v6.exchangerate-api.com/v6/6dc698c520fcc156065468ba/pair/{base_currency}/{target_currency}'
    response = requests.get(url)
    
    return response.json()

@tool
def convert(base_currency:float, conversion_rate:float) -> float:
    """this function calculates target currency value from the given base currency value"""
    
    return base_currency * conversion_rate


def extract_code(currency_str):
    match = re.search(r'\((.*?)\)', currency_str)
    return match.group(1) if match else None

currency_options = ["-- Select --"] + currencies

base_currency_full = st.selectbox("Select a base currency:", options=currency_options)
target_currency_full = st.selectbox("Select a target currency:", options=currency_options)


base_currency_code = extract_code(base_currency_full)
target_currency_code = extract_code(target_currency_full)
llm_with_tools = llm.bind_tools([get_conversion_rate, convert])

user_input = int(st.number_input(
    label="User input",
    min_value=1,
    step=1,
    format="%d",
    help="Enter the amount to convert."
))

messages = [HumanMessage(f"What is the conversion factor between {base_currency_code} and {target_currency_code}, and based on that convert {user_input} amount {base_currency_full} to {target_currency_code}")]
ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)
st.write(messages)

if st.button("send"):
    for tool_call in ai_message.tool_calls:
        if tool_call['name'] == "get_conversion_rate":
            tool_message_1 = get_conversion_rate.invoke(tool_call)
            conversion_rate = json.loads(tool_message_1.content)['conversion_rate']
            messages.append(tool_message_1)
            
        if tool_call['name'] == "convert":
            tool_call['args']['conversion_rate'] = conversion_rate
            tool_message_2 = convert.invoke(tool_call)
            messages.append(tool_message_2)

    st.success(llm_with_tools.invoke(messages).content)


# 592a332b39f75d776b614b0c5af7c121
