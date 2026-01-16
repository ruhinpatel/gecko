import gecko
calc = gecko.load_calc("migration/templates/outputs/BetaRaman")

beta = calc.data.get("beta", {})
print("beta keys:", beta.keys())
print("omega shape:", None if not beta else beta["omega"].shape)
print("values shape:", None if not beta else beta["values"].shape)
print("first components:", None if not beta else beta["components"][:5])
# print("beta['omega']:", beta["omega"])
# print("beta['values']", beta["values"])
