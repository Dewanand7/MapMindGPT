import re


CANONICAL_QA = {
    "what is edi": "EDI, or Electronic Data Interchange, is the structured computer-to-computer exchange of business documents between trading partners.",
    "what is an edi 850": "EDI 850 is an ANSI X12 purchase order transaction sent from a buyer to a seller.",
    "what is edi 850": "EDI 850 is an ANSI X12 purchase order transaction sent from a buyer to a seller.",
    "what is an edi 810": "EDI 810 is an ANSI X12 invoice transaction sent by a supplier to request payment.",
    "what is edi 810": "EDI 810 is an ANSI X12 invoice transaction sent by a supplier to request payment.",
    "what is an edi 856": "EDI 856 is an advance ship notice that communicates shipment contents and tracking details.",
    "what is edi 856": "EDI 856 is an advance ship notice that communicates shipment contents and tracking details.",
    "what is an edi 997": "EDI 997 is a functional acknowledgement that confirms whether an EDI transaction was accepted or rejected.",
    "what is edi 997": "EDI 997 is a functional acknowledgement that confirms whether an EDI transaction was accepted or rejected.",
    "what is xpath": "XPath is an expression language used to select nodes and values from XML documents.",
    "example of substring before": "In XPath/XSLT, substring-before(value, delimiter) returns the text before the first occurrence of the delimiter. Example: substring-before($VAR_FCR_NUMBER, '//') returns the part of $VAR_FCR_NUMBER before //.",
    "example of substringbefore": "In XPath/XSLT, substring-before(value, delimiter) returns the text before the first occurrence of the delimiter. Example: substring-before($VAR_FCR_NUMBER, '//') returns the part of $VAR_FCR_NUMBER before //.",
    "substring before example": "In XPath/XSLT, substring-before(value, delimiter) returns the text before the first occurrence of the delimiter. Example: substring-before($VAR_FCR_NUMBER, '//') returns the part of $VAR_FCR_NUMBER before //.",
    "what is format number": "In XSLT/XPath, format-number(value, pattern) formats a numeric value using a pattern. Example: format-number($amount, '#.00') formats the value with two decimal places.",
    "what is formatnumber": "In XSLT/XPath, format-number(value, pattern) formats a numeric value using a pattern. Example: format-number($amount, '#.00') formats the value with two decimal places.",
    "format number example": "Example: format-number($amount, '#.00') returns the numeric value formatted with two decimal places, such as 12.50.",
    "what is normalize space": "In XPath/XSLT, normalize-space(value) removes leading and trailing whitespace and collapses repeated spaces inside the string.",
    "what is translate": "In XPath/XSLT, translate(value, fromChars, toChars) replaces characters from one set with matching characters from another set. Example: translate($code, '.', '+') replaces dots with plus signs.",
    "what is oracle oic": "Oracle Integration Cloud is Oracle's cloud integration platform for connecting applications, services, and data.",
    "what is seeburger bis": "SEEBURGER BIS is a B2B integration platform used for EDI, partner onboarding, routing, and message processing.",
}


def normalize_question(question):
    normalized = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def get_canonical_answer(question):
    normalized = normalize_question(question)
    answer = CANONICAL_QA.get(normalized)
    if answer:
        return answer

    if "substring before" in normalized or "substringbefore" in normalized:
        return CANONICAL_QA["example of substring before"]
    if "format number" in normalized or "formatnumber" in normalized:
        return CANONICAL_QA["what is format number"]
    if "normalize space" in normalized or "normalizespace" in normalized:
        return CANONICAL_QA["what is normalize space"]

    return None
