import sympy as sp
from scipy import integrate

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel

from fastapi import FastAPI
import uvicorn

from dotenv import load_dotenv



#------------------------------------------------------------------------------#

# Load the OLLAMA_BASE_URL variable from .env file
load_dotenv() 


#------------------------------------------------------------------------------#

#Integration function based on Scipy and Sympy
def integrate_expression(expression: str, lower_bound: float, upper_bound: float) -> float:
    """Numerically integrate a symbolic math expression over [lower_bound, upper_bound].

    Args:
        expression: A math expression in terms of x, e.g. "sin(x) * x**2" or "exp(-x**2)".
        lower_bound: Lower integration bound.
        upper_bound: Upper integration bound.

    Returns:
        The numeric value of the definite integral.
    """
    x = sp.symbols("x")

    # Parse the string into a sympy expression (restricted namespace for safety)
    sym_expr = sp.sympify(expression, locals={"x": x})

    # Convert the symbolic expression into a fast numeric function
    f = sp.lambdify(x, sym_expr, modules=["numpy"])

    result, _abs_err = integrate.quad(f, lower_bound, upper_bound)
    return result


#------------------------------------------------------------------------------#

#Agent setup with self-hosted model via Ollama
#I used LLaMA 3.2 in the 3 billion parameter configuration
model = OllamaModel('llama3.2')

agent = Agent(
    model,
    system_prompt=(
        "You are a physics/math assistant. Users will describe real-world "
        "problems in plain language, not as ready-made integrals. Your job is "
        "to:\n"
        "1. Identify the underlying quantity being asked for (distance, area, "
        "work, total accumulated change, etc.).\n"
        "2. Translate it into a definite integral: figure out the correct "
        "integrand as a function of x (e.g. rewrite acceleration -> velocity "
        "-> position reasoning symbolically) and the correct bounds from the "
        "problem's stated values.\n"
        "3. Call the integrate_expression tool to compute the result -- never "
        "compute the integral or do the arithmetic yourself.\n"
        "4. Report the numeric result, and briefly explain the integral you "
        "constructed and why it matches the problem."
    )
)

#Assign the integration function as a tool for the agent to use

@agent.tool_plain
def integrate_expression_tool(expression: str, lower_bound: float, upper_bound: float) -> float:
    """Integrate a symbolic expression f(x) over [lower_bound, upper_bound] and return the value."""
    return integrate_expression(expression, lower_bound, upper_bound)



#------------------------------------------------------------------------------#

#FastAPI endpoint to receive prompts and return the agent's response
#You can try the entpoint opening the documentation URL at 

app=FastAPI()

@app.post("/prompt")
async def integrate_endpoint(prompt: str):
    response = await agent.run(prompt)
    return response



#------------------------------------------------------------------------------#

#Run the server

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
