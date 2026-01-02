import api from "./api"

export interface User {
  id: string
  email: string
  full_name: string
  is_active: boolean
  created_at: string
}

export interface LoginResponse {
  access_token: string
  token_type: string
}

export async function login(email: string, password: string): Promise<LoginResponse> {
  const formData = new URLSearchParams()
  formData.append("username", email)
  formData.append("password", password)

  const response = await api.post<LoginResponse>("/api/v1/auth/login", formData, {
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
  })

  localStorage.setItem("access_token", response.data.access_token)
  return response.data
}

export async function register(email: string, password: string, fullName: string): Promise<User> {
  const response = await api.post<User>("/api/v1/auth/register", {
    email,
    password,
    full_name: fullName,
  })

  return response.data
}

export async function getCurrentUser(): Promise<User> {
  const response = await api.get<User>("/api/v1/auth/me")
  return response.data
}

export function logout() {
  localStorage.removeItem("access_token")
  window.location.href = "/login"
}

export function isAuthenticated(): boolean {
  return !!localStorage.getItem("access_token")
}
